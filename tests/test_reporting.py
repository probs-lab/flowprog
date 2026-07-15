"""Tests for flowprog.reporting: symbolic views over raw flow tables.

The pipeline under test: raw flow table (stage 1) -> Report symbolic views
(stage 2) -> evaluation by substitution or compiled function (stage 3).
"""

import logging

import pandas as pd
import pytest
import sympy as sy
from rdflib import URIRef

from flowprog import ModelBuilder, ModelStructure, Process, Object, ElementaryExchange
from flowprog.reporting import (
    Grouping,
    Report,
    evaluate_views,
    lambdify_views,
)

MASS = URIRef("http://qudt.org/vocab/quantitykind/Mass")


def MObject(id, *args, **kwargs):
    return Object(id, MASS, *args, **kwargs)


def MExchange(id, *args, **kwargs):
    return ElementaryExchange(id, MASS, *args, **kwargs)


def build_model():
    processes = [
        Process("Dirty", produces=["out"], consumes=[], exchanges=["CO2", "CH4"]),
        Process("Clean", produces=["out"], consumes=[], exchanges=["CO2"]),
        Process("Use", produces=[], consumes=["out"]),
    ]
    objects = [MObject("out", has_market=True)]
    exchanges = [MExchange("CO2"), MExchange("CH4")]
    builder = ModelBuilder(processes, objects, exchanges)

    demand = sy.Symbol("demand", positive=True)
    builder.add(builder.push_consumption("out", demand))
    builder.add(builder.pull_process_output("Dirty", "out", demand * sy.Rational(3, 4)))
    builder.add(builder.pull_process_output("Clean", "out", demand * sy.Rational(1, 4)))

    recipe = {
        "Dirty": {"produces": {"out": 1.0}, "exchanges": {"CO2": 2.0, "CH4": 0.1}},
        "Clean": {"produces": {"out": 1.0}, "exchanges": {"CO2": 0.5}},
        "Use": {"consumes": {"out": 1.0}},
    }
    model = builder.build(recipe)
    return model, demand


GWP = {"CO2": 1, "CH4": 28}
STAGES = {"upstream": {"Dirty", "Clean"}, "downstream": {"Use"}}

# At demand=100: Y_Dirty=75, Y_Clean=25
# CO2: 75*2 + 25*0.5 = 162.5; CH4: 75*0.1 = 7.5; GWP: 162.5 + 7.5*28 = 372.5
EXPECTED_CO2 = 162.5
EXPECTED_CH4 = 7.5
EXPECTED_GWP = 372.5


class TestGrouping:
    def test_disjoint_groups_build_ok(self):
        mapping = Grouping.build(
            "stage",
            {"upstream": {"Dirty", "Clean"}, "downstream": {"Use"}},
            all_ids={"Dirty", "Clean", "Use"},
        )
        assert mapping == {"Dirty": "upstream", "Clean": "upstream", "Use": "downstream"}

    def test_overlap_raises(self):
        with pytest.raises(ValueError):
            Grouping.build(
                "stage",
                {"a": {"Dirty", "Clean"}, "b": {"Clean"}},
                all_ids={"Dirty", "Clean"},
            )

    def test_unknown_id_in_group_raises(self):
        with pytest.raises(ValueError):
            Grouping.build(
                "stage", {"a": {"DoesNotExist"}}, all_ids={"Dirty", "Clean"}
            )

    def test_remainder_assigned_to_other_and_warns(self, caplog):
        with caplog.at_level(logging.WARNING):
            mapping = Grouping.build(
                "stage", {"upstream": {"Dirty"}}, all_ids={"Dirty", "Clean"}
            )
        assert mapping["Clean"] == "other"
        assert any("other" in rec.message for rec in caplog.records)

    def test_no_warning_when_fully_covered(self, caplog):
        with caplog.at_level(logging.WARNING):
            Grouping.build(
                "stage",
                {"upstream": {"Dirty"}, "downstream": {"Clean"}},
                all_ids={"Dirty", "Clean"},
            )
        assert len(caplog.records) == 0

    def test_complete_builds_explicit_catch_all(self, caplog):
        groups = Grouping.complete({"upstream": {"Dirty"}}, {"Dirty", "Clean"})
        assert groups == {"upstream": {"Dirty"}, "other": {"Clean"}}
        with caplog.at_level(logging.WARNING):
            mapping = Grouping.build("stage", groups, all_ids={"Dirty", "Clean"})
        assert mapping["Clean"] == "other"
        assert len(caplog.records) == 0

    def test_invert_builds_label_to_ids(self):
        groups = Grouping.invert({"Dirty": "upstream", "Clean": "upstream", "Use": "downstream"})
        assert groups == {"upstream": {"Dirty", "Clean"}, "downstream": {"Use"}}


class TestReportConstruction:
    def test_elementary_flows_wraps_raw_table(self):
        model, demand = build_model()
        rep = Report.elementary_flows(model.structure)
        assert {"exchange", "process", "value"} <= set(rep.table.columns)
        # Values are raw symbolic expressions, not numbers
        assert all(isinstance(v, sy.Basic) for v in rep.table["value"])

    def test_table_without_value_column_rejected(self):
        with pytest.raises(ValueError, match="value"):
            Report(pd.DataFrame({"process": ["A"]}))

    def test_accepts_substitute_table(self):
        # Any table of the right shape can be reported on -- reporting does
        # not care where it came from (see test_passthrough for the real
        # pass-through case).
        model, demand = build_model()
        table = model.structure.elementary_flow_table()
        rep = Report.elementary_flows(model.structure, table.iloc[:1])
        assert len(rep.table) == 1


class TestWithGroup:
    def test_adds_label_column(self):
        model, demand = build_model()
        rep = Report.elementary_flows(model.structure).with_group("stage", STAGES, on="process")
        assert set(rep.table["stage"]) <= {"upstream", "downstream"}

    def test_axis_is_explicit_not_inferred(self):
        model, demand = build_model()
        rep = Report.elementary_flows(model.structure).with_group(
            "ghg_kind", {"fossil": {"CO2"}, "non_fossil": {"CH4"}}, on="exchange"
        )
        assert set(rep.table["ghg_kind"]) == {"fossil", "non_fossil"}

    def test_unknown_id_raises_when_universe_known(self):
        model, demand = build_model()
        with pytest.raises(ValueError, match="NotAnId"):
            Report.elementary_flows(model.structure).with_group(
                "stage", {"x": {"NotAnId"}}, on="process"
            )

    def test_group_id_absent_from_table_is_ok(self):
        # "Use" has no exchanges so never appears in the flow table, but it
        # is a valid process id and may be named in a grouping.
        model, demand = build_model()
        rep = Report.elementary_flows(model.structure).with_group("stage", STAGES, on="process")
        assert "downstream" not in set(rep.table["stage"])

    def test_missing_column_raises(self):
        model, demand = build_model()
        with pytest.raises(ValueError, match="no column"):
            Report.elementary_flows(model.structure).with_group("stage", STAGES, on="nope")

    def test_no_universe_validates_disjointness_only(self):
        table = pd.DataFrame({"process": ["A", "B"], "value": [sy.S(1), sy.S(2)]})
        rep = Report(table).with_group("g", {"x": {"A", "SomethingElse"}}, on="process")
        assert list(rep.table["g"]) == ["x", "other"]

    def test_filtered_table_does_not_warn_about_ids_outside_it(self, caplog):
        # The structure's full "object" universe is {elec, water}, but the
        # table was filtered down to just "elec" -- grouping it shouldn't
        # warn about "water" not being covered, since it was never here.
        structure = ModelStructure(
            processes=[
                Process("P1", produces=[], consumes=["elec"]),
                Process("P2", produces=[], consumes=["water"]),
            ],
            objects=[MObject("elec"), MObject("water")],
        )
        rep = Report.consumption(structure).filter(object="elec")
        with caplog.at_level(logging.WARNING):
            rep = rep.with_group("utility", {"ElecReq": {"elec"}}, on="object")
        assert len(caplog.records) == 0
        assert list(rep.table["utility"]) == ["ElecReq"]

    def test_drop_unmatched_filters_out_ungrouped_rows(self, caplog):
        model, demand = build_model()
        with caplog.at_level(logging.WARNING):
            rep = Report.elementary_flows(model.structure).with_group(
                "ghg_kind", {"fossil": {"CO2"}}, on="exchange", drop_unmatched=True
            )
        assert len(caplog.records) == 0
        assert set(rep.table["exchange"]) == {"CO2"}
        assert set(rep.table["ghg_kind"]) == {"fossil"}

    def test_drop_unmatched_still_checks_disjointness(self):
        model, demand = build_model()
        with pytest.raises(ValueError, match="disjoint"):
            Report.elementary_flows(model.structure).with_group(
                "g",
                {"a": {"CO2", "CH4"}, "b": {"CH4"}},
                on="exchange",
                drop_unmatched=True,
            )

    def test_drop_unmatched_still_checks_unknown_ids(self):
        model, demand = build_model()
        with pytest.raises(ValueError, match="NotAnId"):
            Report.elementary_flows(model.structure).with_group(
                "stage", {"x": {"NotAnId"}}, on="process", drop_unmatched=True
            )


class TestCharacterise:
    def test_factors_applied_missing_default_zero(self):
        model, demand = build_model()
        rep = Report.elementary_flows(model.structure).characterise({"CO2": 1}, name="CO2_only")
        total = evaluate_views(model, rep.total(), {demand: 100})
        assert float(total) == pytest.approx(EXPECTED_CO2)
        assert rep.characterisation == "CO2_only"

    def test_scalar_one_is_explicit_unit_characterisation(self):
        model, demand = build_model()
        rep = Report.elementary_flows(model.structure).characterise(1)
        total = evaluate_views(model, rep.total(), {demand: 100})
        assert float(total) == pytest.approx(EXPECTED_CO2 + EXPECTED_CH4)


class TestCommensurabilityGuard:
    def test_uncharacterised_total_over_mixed_exchanges_raises(self):
        model, demand = build_model()
        with pytest.raises(ValueError, match="characterise"):
            Report.elementary_flows(model.structure).total()

    def test_uncharacterised_by_stage_raises(self):
        model, demand = build_model()
        rep = Report.elementary_flows(model.structure).with_group("stage", STAGES, on="process")
        with pytest.raises(ValueError, match="characterise"):
            rep.by("stage")

    def test_by_exchange_is_always_allowed(self):
        model, demand = build_model()
        result = evaluate_views(
            model, Report.elementary_flows(model.structure).by("exchange"), {demand: 100}
        )
        assert float(result["CO2"]) == pytest.approx(EXPECTED_CO2)
        assert float(result["CH4"]) == pytest.approx(EXPECTED_CH4)

    def test_single_exchange_total_allowed(self):
        model, demand = build_model()
        rep = Report.elementary_flows(model.structure).filter(exchange="CO2")
        total = evaluate_views(model, rep.total(), {demand: 100})
        assert float(total) == pytest.approx(EXPECTED_CO2)


class TestViews:
    def test_total_symbolic(self):
        model, demand = build_model()
        total = Report.elementary_flows(model.structure).characterise(GWP, name="GWP").total()
        assert isinstance(total, sy.Basic)
        resolved = evaluate_views(model, total)  # fully symbolic
        assert float(resolved.subs(demand, 100)) == pytest.approx(EXPECTED_GWP)

    def test_by_grouping(self):
        model, demand = build_model()
        rep = Report.elementary_flows(model.structure).with_group("stage", STAGES, on="process")
        result = evaluate_views(model, rep.characterise(GWP).by("stage"), {demand: 100})
        assert isinstance(result, pd.Series)
        assert float(result["upstream"]) == pytest.approx(EXPECTED_GWP)

    def test_by_process(self):
        model, demand = build_model()
        result = evaluate_views(
            model,
            Report.elementary_flows(model.structure).characterise(GWP).by("process"),
            {demand: 100},
        )
        assert float(result["Dirty"]) == pytest.approx(150 + 7.5 * 28)
        assert float(result["Clean"]) == pytest.approx(12.5)

    def test_by_two_keys_multiindex(self):
        model, demand = build_model()
        rep = Report.elementary_flows(model.structure).with_group("stage", STAGES, on="process")
        result = evaluate_views(model, rep.by("stage", "exchange"), {demand: 100})
        assert float(result[("upstream", "CO2")]) == pytest.approx(EXPECTED_CO2)
        assert float(result[("upstream", "CH4")]) == pytest.approx(EXPECTED_CH4)

    def test_by_unknown_column_raises(self):
        model, demand = build_model()
        with pytest.raises(ValueError, match="no column"):
            Report.elementary_flows(model.structure).by("not_a_grouping")

    def test_filter_restricts_rows(self):
        model, demand = build_model()
        rep = Report.elementary_flows(model.structure).filter(process={"Dirty"}).characterise(GWP)
        total = evaluate_views(model, rep.total(), {demand: 100})
        assert float(total) == pytest.approx(150 + 7.5 * 28)


class TestStructuralValues:
    """The flow table and all views are purely structural: a function of the
    model structure alone, holding no evaluable model, backend state, or
    recipe values."""

    def test_structure_attached_and_propagated_through_chaining(self):
        model, demand = build_model()
        rep = Report.elementary_flows(model.structure).characterise(GWP).filter(process={"Dirty"})
        assert rep.structure is model.structure

    def test_accepts_bare_structure(self):
        # No evaluable model needed to build views -- only the structure.
        model, demand = build_model()
        rep = Report.elementary_flows(model.structure)
        total = rep.characterise(GWP).total()
        assert float(evaluate_views(model, total, {demand: 100})) == pytest.approx(
            EXPECTED_GWP
        )

    def test_values_use_structural_symbols(self):
        model, demand = build_model()
        table = Report.elementary_flows(model.structure).table
        Y, B = model.structure.Y, model.structure.B
        for value in table["value"]:
            bases = {a.base for a in value.atoms(sy.Indexed)}
            assert bases == {Y, B}

    def test_declared_cell_without_recipe_value_persists(self):
        # "Clean" declares CH4 but this recipe gives it no value: the row
        # exists structurally and its value persists as a symbol (like S/U),
        # rather than being silently zeroed.
        processes = [
            Process("Dirty", produces=["out"], consumes=[], exchanges=["CO2"]),
            Process("Clean", produces=["out"], consumes=[], exchanges=["CO2", "CH4"]),
            Process("Use", produces=[], consumes=["out"]),
        ]
        objects = [MObject("out", has_market=True)]
        exchanges = [MExchange("CO2"), MExchange("CH4")]
        builder = ModelBuilder(processes, objects, exchanges)
        demand = sy.Symbol("demand", positive=True)
        builder.add(builder.push_consumption("out", demand))
        builder.add(builder.pull_process_output("Clean", "out", demand))
        model = builder.build(
            {
                "Clean": {"produces": {"out": 1.0}, "exchanges": {"CO2": 0.5}},
                "Use": {"consumes": {"out": 1.0}},
            }
        )
        result = evaluate_views(
            model, Report.elementary_flows(model.structure).by("exchange"), {demand: 100}
        )
        # CH4 for Clean (index 1) has no recipe value, so it stays symbolic.
        ch4 = sy.S(result["CH4"])
        assert not ch4.is_number
        assert model.structure.B[1, 1] in ch4.free_symbols


class TestEvaluateViews:
    def test_dict_of_views_preserves_structure(self):
        model, demand = build_model()
        gwp = Report.elementary_flows(model.structure).characterise(GWP, name="GWP")
        views = {"total": gwp.total(), "by_exchange": gwp.by("exchange")}
        results = evaluate_views(model, views, {demand: 100})
        assert float(results["total"]) == pytest.approx(EXPECTED_GWP)
        assert float(results["by_exchange"]["CH4"]) == pytest.approx(7.5 * 28)

    def test_dataframe_values_evaluated(self):
        model, demand = build_model()
        rep = Report.elementary_flows(model.structure)
        table = evaluate_views(model, rep.table, {demand: 100})
        rows = {(r.exchange, r.process): float(r.value) for r in table.itertuples()}
        assert rows[("CO2", "Dirty")] == pytest.approx(150)


class TestLambdifyViews:
    def test_matches_substitution_path(self):
        model, demand = build_model()
        staged = Report.elementary_flows(model.structure).with_group("stage", STAGES, on="process")
        gwp = staged.characterise(GWP, name="GWP")
        views = {"total": gwp.total(), "by_stage": gwp.by("stage")}

        func = lambdify_views(model, views)
        results = func({demand: 100})
        expected = evaluate_views(model, views, {demand: 100})

        assert results["total"] == pytest.approx(float(expected["total"]))
        pd.testing.assert_index_equal(
            results["by_stage"].index, expected["by_stage"].index
        )
        for key in expected["by_stage"].index:
            assert results["by_stage"][key] == pytest.approx(
                float(expected["by_stage"][key])
            )

    def test_single_expression(self):
        model, demand = build_model()
        total = Report.elementary_flows(model.structure).characterise(GWP).total()
        func = lambdify_views(model, total)
        assert func({demand: 100}) == pytest.approx(EXPECTED_GWP)

    def test_multiindex_series_reassembled(self):
        model, demand = build_model()
        rep = Report.elementary_flows(model.structure).with_group("stage", STAGES, on="process")
        series = rep.by("stage", "exchange")
        func = lambdify_views(model, series)
        result = func({demand: 100})
        assert result[("upstream", "CO2")] == pytest.approx(EXPECTED_CO2)

    def test_compiled_function_reusable_across_values(self):
        model, demand = build_model()
        func = lambdify_views(
            model, Report.elementary_flows(model.structure).characterise(GWP).total()
        )
        assert func({demand: 100}) == pytest.approx(EXPECTED_GWP)
        assert func({demand: 200}) == pytest.approx(2 * EXPECTED_GWP)


class TestProductionConsumption:
    """production()/consumption(): technosphere flow tables reported with the
    same machinery as elementary flows."""

    def test_production_total(self):
        model, demand = build_model()
        rep = Report.production(model.structure).filter(object="out")
        total = evaluate_views(model, rep.total(), {demand: 100})
        assert float(total) == pytest.approx(100)

    def test_consumption_total(self):
        model, demand = build_model()
        rep = Report.consumption(model.structure).filter(object="out")
        total = evaluate_views(model, rep.total(), {demand: 100})
        assert float(total) == pytest.approx(100)

    def test_production_by_process(self):
        model, demand = build_model()
        rep = Report.production(model.structure).filter(object="out")
        result = evaluate_views(model, rep.by("process"), {demand: 100})
        assert float(result["Dirty"]) == pytest.approx(75)
        assert float(result["Clean"]) == pytest.approx(25)

    def test_consumption_by_declared_grouping(self):
        model, demand = build_model()
        rep = (
            Report.consumption(model.structure)
            .filter(object="out")
            .with_group("stage", STAGES, on="process")
        )
        result = evaluate_views(model, rep.by("stage"), {demand: 100})
        assert float(result["downstream"]) == pytest.approx(100)

    def test_production_by_stage_symbolic(self):
        model, demand = build_model()
        rep = (
            Report.production(model.structure)
            .filter(object="out")
            .with_group("stage", STAGES, on="process")
        )
        result = evaluate_views(model, rep.by("stage"))
        assert float(result["upstream"].subs(demand, 100)) == pytest.approx(100)

    def test_filter_limits_processes(self):
        model, demand = build_model()
        rep = Report.production(model.structure).filter(object="out", process={"Dirty"})
        total = evaluate_views(model, rep.total(), {demand: 100})
        assert float(total) == pytest.approx(75)

    def test_lambdify_path(self):
        model, demand = build_model()
        rep = Report.consumption(model.structure).filter(object="out")
        func = lambdify_views(model, rep.total())
        assert func({demand: 100}) == pytest.approx(100)

    def test_consumption_covers_all_objects(self):
        """consumption() (no object_id) is the technosphere analogue of
        elementary_flows(): one row per (object, consuming process), across
        every consumed object -- so groups can span multiple objects, e.g.
        combining several utility inputs into one requirement."""
        structure = ModelStructure(
            processes=[
                Process("P1", produces=[], consumes=["elec"]),
                Process("P2", produces=[], consumes=["green_elec"]),
                Process("P3", produces=[], consumes=["heat"]),
            ],
            objects=[MObject("elec"), MObject("green_elec"), MObject("heat")],
        )
        rep = Report.consumption(structure).with_group(
            "utility",
            {"ElecReq": {"elec", "green_elec"}, "NGReq": {"heat"}},
            on="object",
        )
        result = rep.by("utility")
        assert set(result.index) == {"ElecReq", "NGReq"}
        assert result["ElecReq"] == (
            structure.X[0] * structure.U[0, 0] + structure.X[1] * structure.U[1, 1]
        )
        assert result["NGReq"] == structure.X[2] * structure.U[2, 2]
