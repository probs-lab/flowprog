"""Tests for flowprog.allocation.PassThrough: symbolic reattribution of
pass-through processes' burdens to their direct consumers.

Only the closed-form degenerate case (input-less, single-output, sole-supplier
pass-through processes -- e.g. generated boundary Sources) is implemented;
the general cases raise NotImplementedError with pointers to what is missing.
"""

import pandas as pd
import pytest
import sympy as sy
from rdflib import URIRef

from flowprog import ModelBuilder, Process, Object, ElementaryExchange
from flowprog.allocation import PassThrough
from flowprog.reporting import Report, evaluate_views, lambdify_views

MASS = URIRef("http://qudt.org/vocab/quantitykind/Mass")
ENERGY = URIRef("http://qudt.org/vocab/quantitykind/Energy")

EF_ELEC = sy.Symbol("EF_elec")
DEMAND = sy.Symbol("demand", positive=True)

RECIPE = {
    "SourceOfElec": {
        "produces": {"elec": 1.0},
        "exchanges": {"GHG_up_elec": EF_ELEC},
    },
    "A": {
        "produces": {"prod": 1.0},
        "consumes": {"elec": 0.5},
        "exchanges": {"CO2": 2.0},
    },
    "B": {
        "produces": {"prod": 1.0},
        "consumes": {"elec": 0.25},
    },
    "Use": {"consumes": {"prod": 1.0}},
}


def build_model(dispatch_source=True, recipe=RECIPE, source_exchanges=("GHG_up_elec",)):
    """Two producers (A burden-carrying, B not) both consuming boundary-supplied
    electricity, mirroring the petrochem utility pattern in miniature."""
    processes = [
        Process(
            "SourceOfElec",
            produces=["elec"],
            consumes=[],
            exchanges=list(source_exchanges),
        ),
        Process("A", produces=["prod"], consumes=["elec"], exchanges=["CO2"]),
        Process("B", produces=["prod"], consumes=["elec"]),
        Process("Use", produces=[], consumes=["prod"]),
    ]
    objects = [
        Object("elec", ENERGY, has_market=True),
        Object("prod", MASS, has_market=True),
    ]
    exchanges = [ElementaryExchange("CO2", MASS), ElementaryExchange("GHG_up_elec", MASS)]
    builder = ModelBuilder(processes, objects, exchanges)

    builder.add(builder.push_consumption("prod", DEMAND))
    builder.add(
        builder.pull_process_output(
            "A", "prod", DEMAND * sy.Rational(3, 4), until_objects=["elec"]
        )
    )
    builder.add(
        builder.pull_process_output(
            "B", "prod", DEMAND * sy.Rational(1, 4), until_objects=["elec"]
        )
    )
    if dispatch_source:
        builder.add(
            builder.pull_process_output(
                "SourceOfElec", "elec", builder.object_production_deficit("elec")
            )
        )
    return builder.build(recipe)


VALUES = {DEMAND: 100, EF_ELEC: 0.2}

# At demand=100: Y_A = 75, Y_B = 25; elec use = 75*0.5 + 25*0.25 = 43.75
EXPECTED_A_ELEC = 75 * 0.5 * 0.2  # 7.5
EXPECTED_B_ELEC = 25 * 0.25 * 0.2  # 1.25
EXPECTED_A_CO2 = 75 * 2.0


def evaluate(model, expr):
    return float(model.eval(sy.S(expr), VALUES))


class TestPassThroughFlows:
    def test_source_burden_reattributed_to_consumers(self):
        model = build_model()
        pt = PassThrough(model.structure, ["SourceOfElec"])
        flows = pt.elementary_flows()

        rows = {
            (r.exchange, r.process): evaluate(model, r.value)
            for r in flows.itertuples()
        }
        assert rows[("GHG_up_elec", "A")] == pytest.approx(EXPECTED_A_ELEC)
        assert rows[("GHG_up_elec", "B")] == pytest.approx(EXPECTED_B_ELEC)
        # Direct burdens stay where they are
        assert rows[("CO2", "A")] == pytest.approx(EXPECTED_A_CO2)
        # The pass-through process's own rows are gone
        assert "SourceOfElec" not in set(flows["process"])

    def test_via_column_records_provenance(self):
        model = build_model()
        flows = PassThrough(model.structure, ["SourceOfElec"]).elementary_flows()
        via = {
            (r.exchange, r.process): r.via for r in flows.itertuples()
        }
        assert via[("GHG_up_elec", "A")] == "SourceOfElec"
        assert pd.isna(via[("CO2", "A")])

    def test_totals_preserved_when_market_balances(self):
        model = build_model()
        pt = PassThrough(model.structure, ["SourceOfElec"])

        def total(table):
            return sum(evaluate(model, v) for v in table["value"])

        assert total(pt.elementary_flows()) == pytest.approx(
            total(model.structure.elementary_flow_table())
        )

    def test_residuals_zero_when_market_balances(self):
        model = build_model()
        residuals = PassThrough(model.structure, ["SourceOfElec"]).residuals()
        assert len(residuals) == 1
        assert evaluate(model, residuals["value"].iloc[0]) == pytest.approx(0)

    def test_residuals_nonzero_when_market_unbalanced(self):
        # Without the dispatch step the source never runs: everything that
        # would have been redistributed shows up as (negative) residual.
        model = build_model(dispatch_source=False)
        residuals = PassThrough(model.structure, ["SourceOfElec"]).residuals()
        assert evaluate(model, residuals["value"].iloc[0]) == pytest.approx(
            -(EXPECTED_A_ELEC + EXPECTED_B_ELEC)
        )

    def test_burden_less_pass_through_is_noop(self):
        # A pass-through process that declares no exchanges carries no burden,
        # so reattribution adds nothing (structural: sparsity is declared).
        recipe = {
            **RECIPE,
            "SourceOfElec": {"produces": {"elec": 1.0}},
        }
        model = build_model(recipe=recipe, source_exchanges=())
        flows = PassThrough(model.structure, ["SourceOfElec"]).elementary_flows()
        assert set(flows["exchange"]) == {"CO2"}

    def test_non_unit_supply_coefficient_divides_intensity(self):
        # S = 2 units of elec per unit activity: burden per unit elec halves.
        recipe = {
            **RECIPE,
            "SourceOfElec": {
                "produces": {"elec": 2.0},
                "exchanges": {"GHG_up_elec": EF_ELEC},
            },
        }
        model = build_model(recipe=recipe)
        flows = PassThrough(model.structure, ["SourceOfElec"]).elementary_flows()
        rows = {
            (r.exchange, r.process): evaluate(model, r.value)
            for r in flows.itertuples()
        }
        assert rows[("GHG_up_elec", "A")] == pytest.approx(EXPECTED_A_ELEC / 2)


class TestPassThroughReporting:
    """Reporting composes with pass-through by table substitution: the
    reattributed table is dropped into `elementary_flows()` in place of the
    model's own, and everything downstream is identical."""

    def test_by_stage_and_exchange(self):
        model = build_model()
        pt = PassThrough(model.structure, ["SourceOfElec"])
        rep = Report.elementary_flows(model.structure, pt.elementary_flows()).with_group(
            "stage",
            {"making": {"A", "B"}, "boundary": {"SourceOfElec"}, "use": {"Use"}},
            on="process",
        )
        result = evaluate_views(model, rep.by("stage", "exchange"), VALUES)
        assert float(result[("making", "GHG_up_elec")]) == pytest.approx(
            EXPECTED_A_ELEC + EXPECTED_B_ELEC
        )
        assert float(result[("making", "CO2")]) == pytest.approx(EXPECTED_A_CO2)
        # Nothing left attributed to the pass-through process itself
        assert ("boundary", "GHG_up_elec") not in result

    def test_by_process(self):
        model = build_model()
        pt = PassThrough(model.structure, ["SourceOfElec"])
        rep = Report.elementary_flows(model.structure, pt.elementary_flows()).characterise(1)
        by_process = evaluate_views(model, rep.by("process"), VALUES)
        assert float(by_process["A"]) == pytest.approx(
            EXPECTED_A_CO2 + EXPECTED_A_ELEC
        )

    def test_by_via_column_attributes_received_burden(self):
        # The `via` column from PassThrough is just another groupable column.
        model = build_model()
        pt = PassThrough(model.structure, ["SourceOfElec"])
        rep = Report.elementary_flows(model.structure, pt.elementary_flows()).characterise(1)
        by_via = evaluate_views(model, rep.by("via"), VALUES)
        assert float(by_via["SourceOfElec"]) == pytest.approx(
            EXPECTED_A_ELEC + EXPECTED_B_ELEC
        )

    def test_views_resolve_through_lambdify(self):
        model = build_model()
        pt = PassThrough(model.structure, ["SourceOfElec"])
        rep = Report.elementary_flows(model.structure, pt.elementary_flows()).characterise(1)
        total = rep.total()
        func = lambdify_views(model, total)
        expected = evaluate_views(model, total, VALUES)
        assert func(VALUES) == pytest.approx(float(expected))


class TestPassThroughValidation:
    def test_unknown_process_raises(self):
        model = build_model()
        with pytest.raises(ValueError):
            PassThrough(model.structure, ["NoSuchProcess"])

    def test_rule_not_implemented(self):
        model = build_model()
        with pytest.raises(NotImplementedError, match="rule"):
            PassThrough(model.structure, ["SourceOfElec"], rule=object())

    def test_multi_output_not_implemented(self):
        processes = [
            Process("Cracker", produces=["ethylene", "propylene"], consumes=[]),
            Process("Use", produces=[], consumes=["ethylene"]),
        ]
        objects = [
            Object("ethylene", MASS, has_market=True),
            Object("propylene", MASS, has_market=True),
        ]
        builder = ModelBuilder(processes, objects, [ElementaryExchange("CO2", MASS)])
        model = builder.build(
            {"Cracker": {"produces": {"ethylene": 0.6, "propylene": 0.4}}}
        )
        with pytest.raises(NotImplementedError, match="allocation rule"):
            PassThrough(model.structure, ["Cracker"])

    def test_inputs_not_implemented(self):
        model = build_model()
        with pytest.raises(NotImplementedError, match="inputs"):
            PassThrough(model.structure, ["A"])  # A consumes elec

    def test_co_supplied_object_not_implemented(self):
        processes = [
            Process("Source1", produces=["elec"], consumes=[]),
            Process("Source2", produces=["elec"], consumes=[]),
            Process("Use", produces=[], consumes=["elec"]),
        ]
        objects = [Object("elec", ENERGY, has_market=True)]
        builder = ModelBuilder(processes, objects, [ElementaryExchange("CO2", MASS)])
        model = builder.build(
            {
                "Source1": {"produces": {"elec": 1.0}},
                "Source2": {"produces": {"elec": 1.0}},
                "Use": {"consumes": {"elec": 1.0}},
            }
        )
        with pytest.raises(NotImplementedError, match="market-share"):
            PassThrough(model.structure, ["Source1"])
