"""Reporting on top of the NumPyro backend.

Validates that the reporting design is backend-agnostic. The three pipeline
stages decouple cleanly from the sympy backend:

1. Flow table: ``ModelStructure.elementary_flow_table()`` builds the table
   from *structural* symbols (``Y[j] * B[e, j]``) -- shared by every backend,
   independent of how each represents accumulated state.
2. Symbolic views: `Report` is pure pandas + sympy, so grouping/
   characterisation/aggregation code is identical for both backends.
3. Evaluation: anything with ``eval(expr, values)`` resolves the views --
   here a ``NumpyroState`` from a forward model run, in place of
   ``SympyModel``'s substitution.

Because the numpyro table's values are structural (not baked accumulated
expressions), the *same* view expressions also evaluate through
``SympyModel.eval`` -- checked below.
"""

import numpyro.handlers as handlers
import pytest
import sympy as sy
from rdflib import URIRef

from flowprog import ModelBuilder, Process, Object, ElementaryExchange
from flowprog.backends.numpyro import NumpyroModel
from flowprog.reporting import Report, evaluate_views

MASS = URIRef("http://qudt.org/vocab/quantitykind/Mass")

DEMAND = sy.Symbol("demand", positive=True)

RECIPE = {
    "Dirty": {"produces": {"out": 1.0}, "exchanges": {"CO2": 2.0, "CH4": 0.1}},
    "Clean": {"produces": {"out": 1.0}, "exchanges": {"CO2": 0.5}},
    "Use": {"consumes": {"out": 1.0}},
}

GWP = {"CO2": 1, "CH4": 28}
STAGES = {"upstream": {"Dirty", "Clean"}, "downstream": {"Use"}}

# At demand=100: Y_Dirty=75, Y_Clean=25
# CO2: 75*2 + 25*0.5 = 162.5; CH4: 75*0.1 = 7.5; GWP: 162.5 + 7.5*28 = 372.5
EXPECTED_CO2 = 162.5
EXPECTED_CH4 = 7.5
EXPECTED_GWP = 372.5


def build_builder():
    """Same model as test_reporting.build_model(), kept as a builder so it
    can be compiled through either backend."""
    processes = [
        Process("Dirty", produces=["out"], consumes=[], exchanges=["CO2", "CH4"]),
        Process("Clean", produces=["out"], consumes=[], exchanges=["CO2"]),
        Process("Use", produces=[], consumes=["out"]),
    ]
    objects = [Object("out", MASS, has_market=True)]
    exchanges = [ElementaryExchange("CO2", MASS), ElementaryExchange("CH4", MASS)]
    builder = ModelBuilder(processes, objects, exchanges)

    builder.add(builder.push_consumption("out", DEMAND))
    builder.add(
        builder.pull_process_output("Dirty", "out", DEMAND * sy.Rational(3, 4))
    )
    builder.add(
        builder.pull_process_output("Clean", "out", DEMAND * sy.Rational(1, 4))
    )
    return builder


def numpyro_state(builder, params):
    """Compile through the numpyro backend and run the forward model once,
    returning the resulting state (parameters bound, X/Y accumulated)."""
    compiled = NumpyroModel.from_steps(
        builder._steps, builder.structure, recipe_data=RECIPE
    )
    with handlers.seed(rng_seed=0):
        state = compiled(params)
    return compiled, state


class TestStructuralFlowTable:
    """The elementary-flow table lives on ModelStructure and is shared by
    every backend -- there is no backend-specific to_elementary_flows."""

    def test_table_is_structure_level(self):
        builder = build_builder()
        # Same structure -> identical table regardless of backend.
        sympy_table = builder.build(RECIPE).structure.elementary_flow_table()
        numpyro_table = build_builder().structure.elementary_flow_table()
        assert list(numpyro_table.columns) == list(sympy_table.columns)
        assert set(zip(numpyro_table["exchange"], numpyro_table["process"])) == set(
            zip(sympy_table["exchange"], sympy_table["process"])
        )

    def test_values_are_structural(self):
        # Values reference the structural Y[j] symbols, not accumulated
        # expressions -- accumulation only happens during a forward run.
        table = build_builder().structure.elementary_flow_table()
        Y = build_builder().structure.Y
        for value in table["value"]:
            assert any(a.base == Y for a in value.atoms(sy.Indexed))


class TestReportingOnNumpyro:
    """The same Report machinery as the sympy backend, evaluated against a
    forward-run NumpyroState instead of by symbolic substitution."""

    def _report(self):
        builder = build_builder()
        compiled, state = numpyro_state(builder, {"demand": 100.0})
        rep = Report.elementary_flows(compiled.structure).with_group(
            "stage", STAGES, on="process"
        )
        return rep, state

    def test_characterised_total(self):
        rep, state = self._report()
        total = rep.characterise(GWP, name="GWP").total()
        assert float(state.eval(total)) == pytest.approx(EXPECTED_GWP)

    def test_by_exchange_via_evaluate_views(self):
        rep, state = self._report()
        result = evaluate_views(state, rep.by("exchange"))
        assert float(result["CO2"]) == pytest.approx(EXPECTED_CO2)
        assert float(result["CH4"]) == pytest.approx(EXPECTED_CH4)

    def test_by_stage_and_exchange(self):
        rep, state = self._report()
        result = evaluate_views(state, rep.by("stage", "exchange"))
        assert float(result[("upstream", "CO2")]) == pytest.approx(EXPECTED_CO2)
        assert float(result[("upstream", "CH4")]) == pytest.approx(EXPECTED_CH4)

    def test_commensurability_guard_applies(self):
        # Stage-2 behaviour is backend-independent: same guard as sympy.
        rep, state = self._report()
        with pytest.raises(ValueError, match="characterise"):
            rep.total()

    def test_grouping_typo_guard_applies(self):
        builder = build_builder()
        compiled, _ = numpyro_state(builder, {"demand": 100.0})
        with pytest.raises(ValueError, match="NotAnId"):
            Report.elementary_flows(compiled.structure).with_group(
                "stage", {"x": {"NotAnId"}}, on="process"
            )

    def test_state_reflects_its_parameters(self):
        # A different forward run evaluates the same views at its own
        # parameter point -- no recompilation of the views needed.
        builder = build_builder()
        _, state200 = numpyro_state(builder, {"demand": 200.0})
        rep = Report.elementary_flows(builder.structure)
        total = rep.characterise(GWP).total()
        assert float(state200.eval(total)) == pytest.approx(2 * EXPECTED_GWP)


class TestViewsPortableAcrossBackends:
    def test_structural_views_evaluate_on_both_backends(self):
        # Views built from the structural (numpyro) flow table contain only
        # structural symbols + recipe values, so the *same expressions*
        # resolve through SympyModel.eval and NumpyroState.eval alike.
        builder = build_builder()
        compiled, state = numpyro_state(builder, {"demand": 100.0})
        sympy_model = builder.build(RECIPE)

        views = (
            Report.elementary_flows(compiled.structure)
            .with_group("stage", STAGES, on="process")
            .characterise(GWP, name="GWP")
            .by("stage", "exchange")
        )

        via_numpyro = evaluate_views(state, views)
        via_sympy = evaluate_views(sympy_model, views, {DEMAND: 100})

        assert list(via_numpyro.index) == list(via_sympy.index)
        for key in views.index:
            assert float(via_numpyro[key]) == pytest.approx(float(via_sympy[key]))

    def test_matches_sympy_backend_report(self):
        # End-to-end agreement: each backend's own table, same Report code,
        # same numbers.
        builder = build_builder()
        compiled, state = numpyro_state(builder, {"demand": 100.0})
        sympy_model = builder.build(RECIPE)

        def gwp_by_stage(rep):
            return rep.with_group("stage", STAGES, on="process").characterise(
                GWP
            ).by("stage")

        via_numpyro = evaluate_views(state, gwp_by_stage(Report.elementary_flows(compiled.structure)))
        via_sympy = evaluate_views(
            sympy_model, gwp_by_stage(Report.elementary_flows(sympy_model.structure)), {DEMAND: 100}
        )
        assert float(via_numpyro["upstream"]) == pytest.approx(
            float(via_sympy["upstream"])
        )
