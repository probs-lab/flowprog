"""Tests for flowprog.allocation: attributional average burdens (mu, beta)."""

import numpy as np
import pandas as pd
import pytest
import sympy as sy
from rdflib import URIRef

from flowprog import ModelBuilder, Process, Object, ElementaryExchange
from flowprog.boundary_processes import Import, Export, add_boundary_processes
from flowprog.allocation import (
    AllocatedSystem,
    ByProperty,
    Excluding,
    Fixed,
    ByValue,
    Rules,
    Scope,
)

MASS = URIRef("http://qudt.org/vocab/quantitykind/Mass")


def MObject(id, *args, **kwargs):
    return Object(id, MASS, *args, **kwargs)


def MExchange(id, *args, **kwargs):
    return ElementaryExchange(id, MASS, *args, **kwargs)


# ============================================================================
# Toy analytic model (acceptance test 2 / 2a):
#
# ImportOre --Ore--> Smelting --MetalA--> (final demand, boundary output)
#                        |     --MetalB--> ExportMetalB (burden-carrying export)
#                        (direct CO2)
#
# Hand-computed (see derivation in session notes):
#   Y_ImportOre=40, Y_Smelting=20, Y_ExportMetalB=8
#   Mass allocation:   mu_MetalA = mu_MetalB = 25, beta_ExportMetalB = 27
#   Manual (0.8/0.2):  mu_MetalA = 33.3333, mu_MetalB = 12.5, beta_ExportMetalB = 14.5
#   Conservation (both rules): sum(mu_i * boundary_output_i) + sink_terms == total direct burden
# ============================================================================


def build_toy_model():
    processes = [
        Process(
            "Smelting",
            produces=["MetalA", "MetalB"],
            consumes=["Ore"],
            exchanges=["CO2"],
        ),
    ]
    objects = [
        MObject("Ore", has_market=True),
        MObject("MetalA", has_market=True),
        MObject("MetalB", has_market=True),
    ]
    exchanges = [MExchange("CO2")]
    from flowprog import ModelStructure

    structure = ModelStructure(processes, objects, exchanges)
    structure, fragment_import = add_boundary_processes(
        structure, [Import("Ore", exchanges={"CO2": 10})]
    )
    structure, fragment_export = add_boundary_processes(
        structure, [Export("MetalB", exchanges={"CO2": 2})]
    )

    builder = ModelBuilder.from_structure(structure)

    metal_a_demand = sy.Symbol("metal_a_demand", positive=True)
    builder.add(
        builder.pull_process_output(
            "Smelting", "MetalA", metal_a_demand
        )
    )
    builder.add(
        builder.push_process_input(
            "ExportsOfMetalB", "MetalB", builder.object_consumption_deficit("MetalB")
        )
    )

    recipe = {
        "Smelting": {
            "consumes": {"Ore": 2.0},
            "produces": {"MetalA": 0.6, "MetalB": 0.4},
            "exchanges": {"CO2": 5.0},
        },
        **fragment_import,
        **fragment_export,
    }
    model = builder.build(recipe)
    return model, metal_a_demand


SMELTING = Process("Smelting", produces=["MetalA", "MetalB"], consumes=["Ore"])
FLOWS = {"MetalA": 0.6, "MetalB": 0.4}


class TestRules:
    def test_mass_weights_are_the_flows(self):
        assert ByValue().weights(SMELTING, FLOWS) == FLOWS

    def test_by_property_scales_flows_by_property(self):
        rule = ByProperty({"MetalA": 3.0, "MetalB": 1.0})
        assert rule.weights(SMELTING, FLOWS) == pytest.approx(
            {"MetalA": 1.8, "MetalB": 0.4}
        )

    def test_by_property_missing_value_raises(self):
        rule = ByProperty({"MetalA": 3.0})
        with pytest.raises(ValueError, match="MetalB.*Smelting"):
            rule.weights(SMELTING, FLOWS)

    def test_fixed_ignores_the_flows(self):
        rule = Fixed({"MetalA": 0.8, "MetalB": 0.2})
        assert rule.weights(SMELTING, FLOWS) == {"MetalA": 0.8, "MetalB": 0.2}

    def test_fixed_missing_value_raises(self):
        rule = Fixed({"MetalA": 0.8})
        with pytest.raises(ValueError, match="MetalB.*Smelting"):
            rule.weights(SMELTING, FLOWS)

    def test_excluding_zeroes_named_objects(self):
        rule = Excluding({"MetalB"}, ByValue())
        assert rule.weights(SMELTING, FLOWS) == {"MetalA": 0.6, "MetalB": 0.0}

    def test_excluding_does_not_ask_the_inner_rule_about_excluded_objects(self):
        rule = Excluding({"MetalB"}, ByProperty({"MetalA": 3.0}))
        assert rule.weights(SMELTING, FLOWS) == pytest.approx(
            {"MetalA": 1.8, "MetalB": 0.0}
        )

    def test_rules_dispatches_by_process(self):
        rule = Rules(default=ByValue(), by_process={"Smelting": Fixed({"MetalA": 1.0, "MetalB": 0.0})})
        assert rule.weights(SMELTING, FLOWS) == {"MetalA": 1.0, "MetalB": 0.0}
        other = Process("Refining", produces=["MetalA"], consumes=["Ore"])
        assert rule.weights(other, {"MetalA": 1.0}) == {"MetalA": 1.0}

    def test_rules_assignments_lists_multi_output_processes(self):
        model, _ = build_toy_model()
        rule = Rules(default=ByValue(), by_process={"Smelting": Fixed({})})
        assignments = rule.assignments(model.structure).set_index("process")
        assert list(assignments.index) == ["Smelting"]
        assert assignments.loc["Smelting", "rule"] == "Fixed"

    def test_unknown_ids_are_rejected(self):
        model, demand = build_toy_model()
        with pytest.raises(ValueError, match="Smelteng"):
            AllocatedSystem(model, {demand: 12}, Rules(ByValue(), {"Smelteng": ByValue()}))
        with pytest.raises(ValueError, match="MetalC"):
            AllocatedSystem(model, {demand: 12}, Excluding({"MetalC"}, ByValue()))


class TestToyAnalyticMassAllocation:
    def test_mu_and_beta_match_hand_computation(self):
        model, demand = build_toy_model()
        values = {demand: 12}
        result = AllocatedSystem(model, values, ByValue())

        # Demand 12 --> Y_Smelting = 12/0.6 = 20.
        # Ore production = 20 * 2 = 40.
        #
        # CO2 from Ore imports = 40 * 10 = 400, intensity 10.
        #
        # CO2 from Smelting = 5 * 20 = 100, so intensity = (400 + 100)/20 = 25
        # for the process.
        #
        # Object intensities are then 400/40=10 for Ore and 25*0.6/0.6 = 25
        # (allocated) for MetalA, and similarly for MetalB.

        mu = result.object_intensities
        assert mu.loc["Ore", "CO2"] == pytest.approx(10)
        assert mu.loc["MetalA", "CO2"] == pytest.approx(25)
        assert mu.loc["MetalB", "CO2"] == pytest.approx(25)

        beta = result.process_intensities
        assert beta.loc["ImportsOfOre", "CO2"] == pytest.approx(10)
        assert beta.loc["Smelting", "CO2"] == pytest.approx(25)
        assert beta.loc["ExportsOfMetalB", "CO2"] == pytest.approx(27)

    def test_conservation_holds(self):
        model, demand = build_toy_model()
        values = {demand: 12}
        result = AllocatedSystem(model, values, ByValue())
        assert result.check_conservation(atol=1e-6)

    def test_export_is_boundary_output_not_cradle_to_gate(self):
        """Sink/export burdens are excluded from cradle-to-gate mu (scope
        excludes the export process) but present in the whole-system total."""
        model, demand = build_toy_model()
        values = {demand: 12}

        whole_system = AllocatedSystem(model, values, ByValue())
        cradle_to_gate = AllocatedSystem(
            model,
            values,
            ByValue(),
            scope=Scope(excluded_processes=frozenset({"ExportsOfMetalB"})),
        )

        # Whole-system: export's direct burden is reachable (present in totals)
        assert whole_system.process_intensities.loc["ExportsOfMetalB", "CO2"] == pytest.approx(27)
        # Cradle-to-gate: export process is out of scope entirely
        assert "ExportsOfMetalB" not in cradle_to_gate.process_intensities.index or np.isnan(
            cradle_to_gate.process_intensities.loc["ExportsOfMetalB", "CO2"]
        )
        # MetalB's own cradle-to-gate mu is unaffected (excludes downstream export burden)
        assert cradle_to_gate.object_intensities.loc["MetalB", "CO2"] == pytest.approx(25)
        assert cradle_to_gate.check_conservation(atol=1e-6)


class TestToyAnalyticFixedWeights:
    def test_mu_and_beta_match_hand_computation(self):
        model, demand = build_toy_model()
        values = {demand: 12}
        rule = Rules(ByValue(), {"Smelting": Fixed({"MetalA": 0.8, "MetalB": 0.2})})
        result = AllocatedSystem(model, values, rule)

        mu = result.object_intensities
        assert mu.loc["MetalA", "CO2"] == pytest.approx(100 / 3)
        assert mu.loc["MetalB", "CO2"] == pytest.approx(12.5)

        beta = result.process_intensities
        assert beta.loc["ExportsOfMetalB", "CO2"] == pytest.approx(14.5)

    def test_conservation_holds(self):
        model, demand = build_toy_model()
        values = {demand: 12}
        rule = Rules(ByValue(), {"Smelting": Fixed({"MetalA": 0.8, "MetalB": 0.2})})
        result = AllocatedSystem(model, values, rule)
        assert result.check_conservation(atol=1e-6)


class TestNoDefaultRule:
    def test_uncovered_multi_output_process_raises(self):
        model, demand = build_toy_model()
        values = {demand: 12}
        rule = Fixed({})  # no weights for anything
        with pytest.raises(ValueError, match="Smelting"):
            AllocatedSystem(model, values, rule)


class TestZeroWeightCutoffRecorded:
    def test_zero_weight_nonzero_flow_recorded_in_meta(self):
        model, demand = build_toy_model()
        values = {demand: 12}
        rule = Rules(ByValue(), {"Smelting": Fixed({"MetalA": 1.0, "MetalB": 0.0})})
        result = AllocatedSystem(model, values, rule)
        assert ("Smelting", "MetalB") in result.cutoffs


class TestSupplyShares:
    def test_supply_shares_sum_to_one_per_object(self):
        model, demand = build_toy_model()
        values = {demand: 12}
        result = AllocatedSystem(model, values, ByValue())
        sigma = result.supply_shares
        for obj_id in ("Ore", "MetalA", "MetalB"):
            total = sigma[sigma["object"] == obj_id]["sigma"].sum()
            assert total == pytest.approx(1.0)


class TestZeroSupplyObject:
    def test_nan_does_not_propagate_to_unrelated_objects(self):
        processes = [
            Process("P1", produces=["out1"], consumes=[], exchanges=["CO2"]),
            Process("P2", produces=["out2"], consumes=[], exchanges=["CO2"]),
        ]
        objects = [MObject("out1", has_market=True), MObject("out2", has_market=True)]
        exchanges = [MExchange("CO2")]
        builder = ModelBuilder(processes, objects, exchanges)
        builder.add({builder.Y[0]: 0, builder.Y[1]: 10})
        model = builder.build(
            {
                "P1": {"produces": {"out1": 1.0}, "exchanges": {"CO2": 5.0}},
                "P2": {"produces": {"out2": 1.0}, "exchanges": {"CO2": 3.0}},
            }
        )
        result = AllocatedSystem(model, {}, ByValue())
        assert np.isnan(result.object_intensities.loc["out1", "CO2"])
        assert result.object_intensities.loc["out2", "CO2"] == pytest.approx(3.0)


class TestNegativeMuNotClipped:
    def test_negative_B_gives_negative_mu(self):
        processes = [Process("P1", produces=["out"], consumes=[], exchanges=["CO2"])]
        objects = [MObject("out", has_market=True)]
        exchanges = [MExchange("CO2")]
        builder = ModelBuilder(processes, objects, exchanges)
        builder.add({builder.Y[0]: 10})
        model = builder.build(
            {"P1": {"produces": {"out": 1.0}, "exchanges": {"CO2": -3.5}}}
        )
        result = AllocatedSystem(model, {}, ByValue())
        assert result.object_intensities.loc["out", "CO2"] == pytest.approx(-3.5)


class TestLinearModelEquivalence:
    """Acceptance test 3: for a purely linear model, mu_i equals the elementary
    totals from pulling a unit demand of i."""

    def test_mu_matches_unit_pull(self):
        processes = [
            Process("MakeMid", produces=["mid"], consumes=["raw"]),
            Process("MakeOut", produces=["out"], consumes=["mid"], exchanges=["CO2"]),
        ]
        objects = [
            MObject("raw", has_market=True),
            MObject("mid", has_market=True),
            MObject("out", has_market=True),
        ]
        exchanges = [MExchange("CO2")]
        structure_builder = ModelBuilder(processes, objects, exchanges)
        from flowprog import ModelStructure

        structure = ModelStructure(processes, objects, exchanges)
        structure, fragment = add_boundary_processes(
            structure, [Import("raw", exchanges={"CO2": 7.0})]
        )
        builder = ModelBuilder.from_structure(structure)

        demand = sy.Symbol("demand", positive=True)
        builder.add(builder.pull_production("out", demand, until_objects=[]))

        recipe = {
            "MakeMid": {"consumes": {"raw": 2.0}, "produces": {"mid": 1.0}},
            "MakeOut": {"consumes": {"mid": 3.0}, "produces": {"out": 1.0}, "exchanges": {"CO2": 1.5}},
            **fragment,
        }
        model = builder.build(recipe)

        result = AllocatedSystem(model, {demand: 1}, ByValue())
        mu_out = result.object_intensities.loc["out", "CO2"]

        # Compare to pulling a unit demand of "out" directly and summing elementary flows
        builder2 = ModelBuilder.from_structure(structure)
        builder2.add(builder2.pull_production("out", sy.S.One, until_objects=[]))
        model2 = builder2.build(recipe)
        total_co2 = model2.eval(model2.structure.ElementaryBalance[0])

        assert float(mu_out) == pytest.approx(float(total_co2))


class TestRecyclateLoop:
    """Acceptance test 8: recyclate loop with cut-off disabled solves and conserves."""

    def _model(self):
        processes = [
            Process("Virgin", produces=["Product"], consumes=["Raw"], exchanges=["CO2"]),
            Process("Use", produces=["Waste"], consumes=["Product"]),
            Process(
                "Recycling", produces=["Product"], consumes=["Waste"], exchanges=["CO2"]
            ),
        ]
        objects = [
            MObject("Raw", has_market=True),
            MObject("Product", has_market=True),
            MObject("Waste", has_market=True),
        ]
        exchanges = [MExchange("CO2")]
        from flowprog import ModelStructure

        structure = ModelStructure(processes, objects, exchanges)
        structure, fragment = add_boundary_processes(
            structure, [Import("Raw", exchanges={"CO2": 10.0})]
        )
        builder = ModelBuilder.from_structure(structure)

        demand = sy.Symbol("demand", positive=True)
        # Half of product demand met by recycling, half by virgin production
        builder.add(
            builder.pull_process_output("Recycling", "Product", demand * sy.Rational(1, 2))
        )
        builder.add(
            builder.pull_process_output("Virgin", "Product", demand * sy.Rational(1, 2))
        )
        builder.add(builder.push_consumption("Product", demand))
        builder.add(
            builder.push_process_input(
                "Recycling", "Waste", builder.object_consumption_deficit("Waste")
            )
        )
        builder.add(
            builder.pull_process_output(
                "ImportsOfRaw", "Raw", builder.object_production_deficit("Raw")
            )
        )

        recipe = {
            "Virgin": {"consumes": {"Raw": 1.0}, "produces": {"Product": 1.0}, "exchanges": {"CO2": 2.0}},
            "Use": {"consumes": {"Product": 1.0}, "produces": {"Waste": 1.0}},
            "Recycling": {"consumes": {"Waste": 1.0}, "produces": {"Product": 1.0}, "exchanges": {"CO2": 0.5}},
            **fragment,
        }
        model = builder.build(recipe)
        return model, demand

    def test_cutoff_breaks_loop_and_conserves(self):
        model, demand = self._model()
        scope = Scope(waste_objects=frozenset({"Waste"}), waste_input_burden="cutoff")
        result = AllocatedSystem(model, {demand: 100}, ByValue(), scope=scope)
        assert result.check_conservation(atol=1e-6)
        # Recycled product's mu should be lower than virgin-only mu (cheaper: no Raw import burden)
        assert result.process_intensities.loc["Recycling", "CO2"] == pytest.approx(0.5)

    def test_loop_without_cutoff_solves_and_conserves(self):
        model, demand = self._model()
        scope = Scope(waste_input_burden="propagate")
        result = AllocatedSystem(model, {demand: 100}, ByValue(), scope=scope)
        assert result.check_conservation(atol=1e-6)

    def test_excluding_the_waste_makes_its_producer_a_sink(self):
        """`Use` produces only `Waste`, so excluding `Waste` leaves its burden
        with no bearer: the process is a sink, and its burden is accounted for
        as such rather than attributed to any object."""
        model, demand = self._model()
        result = AllocatedSystem(
            model,
            {demand: 100},
            Excluding({"Waste"}, ByValue()),
            scope=Scope(waste_input_burden="propagate"),
        )

        assert result.sinks == ("Use",)
        assert result.object_intensities.loc["Waste", "CO2"] == pytest.approx(0.0)
        assert result.process_intensities.loc["Recycling", "CO2"] == pytest.approx(0.5)
        assert result.check_conservation(atol=1e-6)

    def test_wastes_shorthand_sets_both_halves(self):
        model, demand = self._model()
        shorthand = AllocatedSystem(model, {demand: 100}, ByValue(), wastes={"Waste"})
        spelled_out = AllocatedSystem(
            model,
            {demand: 100},
            Excluding({"Waste"}, ByValue()),
            scope=Scope(
                waste_objects=frozenset({"Waste"}), waste_input_burden="cutoff"
            ),
        )
        pd.testing.assert_frame_equal(
            shorthand.object_intensities, spelled_out.object_intensities
        )
        assert shorthand.check_conservation(atol=1e-6)

    def test_wastes_and_scope_together_is_an_error(self):
        model, demand = self._model()
        with pytest.raises(ValueError, match="not both"):
            AllocatedSystem(
                model, {demand: 100}, ByValue(), scope=Scope(), wastes={"Waste"}
            )


# ============================================================================
# Stock accumulation: a process taking in more (or less) than it puts out.
#
#   SourceOfOre --Ore--> Warehouse --StoredOre--> Factory --Product-->
#
# Warehouse buys X x 1 ore at mu = 2, ships Y x 1 stored ore, emits 0.5 per
# unit shipped. Running two periods that cancel out (X=15/Y=10 then X=5/Y=10)
# must give each period the same intensities as one combined period with no
# net accumulation (X=Y=20).
# ============================================================================


def build_stock_model(x_level, y_level):
    from flowprog import ModelStructure

    processes = [
        Process("SourceOfOre", produces=["Ore"], consumes=[], exchanges=["CO2"]),
        Process("Warehouse", produces=["StoredOre"], consumes=["Ore"],
                has_stock=True, exchanges=["CO2"]),
        Process("Factory", produces=["Product"], consumes=["StoredOre"]),
    ]
    objects = [MObject("Ore", has_market=True), MObject("StoredOre", has_market=True),
               MObject("Product", has_market=True)]
    structure = ModelStructure(processes, objects, [MExchange("CO2")])
    builder = ModelBuilder.from_structure(structure)
    builder.add({builder.Y[1]: y_level, builder.X[1]: x_level})
    builder.add({builder.Y[0]: builder.X[1] * 1.0})       # ore bought is X * U
    builder.add({builder.X[2]: builder.Y[1] * 1.0, builder.Y[2]: builder.Y[1] * 1.0})
    recipe = {
        "SourceOfOre": {"produces": {"Ore": 1.0}, "exchanges": {"CO2": 2.0}},
        "Warehouse": {"consumes": {"Ore": 1.0}, "produces": {"StoredOre": 1.0},
                      "exchanges": {"CO2": 0.5}},
        "Factory": {"consumes": {"StoredOre": 1.0}, "produces": {"Product": 1.0}},
    }
    return builder.build(recipe)


class TestStockAccumulation:
    PERIODS = {"filling": (15.0, 10.0), "draining": (5.0, 10.0),
               "combined": (20.0, 20.0)}

    def _result(self, key):
        return AllocatedSystem(build_stock_model(*self.PERIODS[key]), {}, ByValue())

    def test_intensities_match_the_combined_period(self):
        combined = self._result("combined").object_intensities
        for key in ("filling", "draining"):
            period = self._result(key).object_intensities
            pd.testing.assert_frame_equal(period, combined)

    def test_stock_burden_is_the_gap_between_what_was_bought_and_charged(self):
        # 5 units of ore either side of what was shipped, at mu[Ore] = 2.
        assert self._result("filling").stock_burden.loc[
            "Warehouse", "CO2"] == pytest.approx(10.0)
        assert self._result("draining").stock_burden.loc[
            "Warehouse", "CO2"] == pytest.approx(-10.0)

    def test_no_stock_burden_when_the_two_sides_match(self):
        assert len(self._result("combined").stock_burden) == 0

    def test_conservation_holds_in_every_period(self):
        for key in self.PERIODS:
            assert self._result(key).check_conservation(atol=1e-9)


class TestConservationCheck:
    def test_undefined_residual_is_a_failure_not_a_pass(self):
        # Only the residuals are needed to check conservation.
        system = AllocatedSystem.__new__(AllocatedSystem)
        system.conservation_residuals = {"CO2": float("nan"), "CH4": 0.0}
        with pytest.raises(AssertionError, match="undefined"):
            system.check_conservation()


class TestSolveRecords:
    def test_rule_and_scope_are_kept(self):
        model, demand = build_toy_model()
        values = {demand: 12}
        result = AllocatedSystem(model, values, ByValue())
        assert isinstance(result.rule, ByValue)
        assert result.scope == Scope()
