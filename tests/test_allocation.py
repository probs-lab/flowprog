"""Tests for flowprog.allocation: attributional average burdens (mu, beta)."""

import numpy as np
import pytest
import sympy as sy
from rdflib import URIRef

from flowprog import ModelBuilder, Process, Object, ElementaryExchange
from flowprog.boundary_processes import Import, Export, add_boundary_processes
from flowprog.allocation import (
    Allocation,
    MassAllocation,
    PropertyAllocation,
    ManualAllocation,
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
            "Smelting", "MetalA", metal_a_demand, until_objects=["Ore"]
        )
    )
    builder.add(
        builder.pull_process_output(
            "ImportsOfOre", "Ore", builder.object_production_deficit("Ore")
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


class TestRules:
    def test_mass_allocation_weight_proportional_to_S(self):
        rule = MassAllocation()
        assert rule.raw_weight("MetalA", "Smelting", 0.6) == 0.6
        assert rule.raw_weight("MetalB", "Smelting", 0.4) == 0.4

    def test_property_allocation_uses_property_value(self):
        rule = PropertyAllocation({"MetalA": 3.0, "MetalB": 1.0})
        assert rule.raw_weight("MetalA", "Smelting", 0.6) == pytest.approx(1.8)

    def test_property_allocation_missing_returns_none(self):
        rule = PropertyAllocation({"MetalA": 3.0})
        assert rule.raw_weight("MetalB", "Smelting", 0.4) is None

    def test_manual_allocation_uses_given_weight(self):
        rule = ManualAllocation({"Smelting": {"MetalA": 0.8, "MetalB": 0.2}})
        assert rule.raw_weight("MetalA", "Smelting", 0.6) == 0.8

    def test_manual_allocation_missing_process_returns_none(self):
        rule = ManualAllocation({})
        assert rule.raw_weight("MetalA", "Smelting", 0.6) is None

    def test_manual_allocation_missing_object_returns_none(self):
        rule = ManualAllocation({"Smelting": {"MetalA": 0.8}})
        assert rule.raw_weight("MetalB", "Smelting", 0.4) is None


class TestToyAnalyticMassAllocation:
    def test_mu_and_beta_match_hand_computation(self):
        model, demand = build_toy_model()
        values = {demand: 12}
        result = Allocation(model, values, MassAllocation()).result

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
        result = Allocation(model, values, MassAllocation()).result
        assert result.check_conservation(atol=1e-6)

    def test_export_is_boundary_output_not_cradle_to_gate(self):
        """Sink/export burdens are excluded from cradle-to-gate mu (scope
        excludes the export process) but present in the whole-system total."""
        model, demand = build_toy_model()
        values = {demand: 12}

        whole_system = Allocation(model, values, MassAllocation()).result
        cradle_to_gate = Allocation(
            model,
            values,
            MassAllocation(),
            scope=Scope(excluded_processes=frozenset({"ExportsOfMetalB"})),
        ).result

        # Whole-system: export's direct burden is reachable (present in totals)
        assert whole_system.process_intensities.loc["ExportsOfMetalB", "CO2"] == pytest.approx(27)
        # Cradle-to-gate: export process is out of scope entirely
        assert "ExportsOfMetalB" not in cradle_to_gate.process_intensities.index or np.isnan(
            cradle_to_gate.process_intensities.loc["ExportsOfMetalB", "CO2"]
        )
        # MetalB's own cradle-to-gate mu is unaffected (excludes downstream export burden)
        assert cradle_to_gate.object_intensities.loc["MetalB", "CO2"] == pytest.approx(25)
        assert cradle_to_gate.check_conservation(atol=1e-6)


class TestToyAnalyticManualAllocation:
    def test_mu_and_beta_match_hand_computation(self):
        model, demand = build_toy_model()
        values = {demand: 12}
        rule = ManualAllocation({"Smelting": {"MetalA": 0.8, "MetalB": 0.2}})
        result = Allocation(model, values, rule).result

        mu = result.object_intensities
        assert mu.loc["MetalA", "CO2"] == pytest.approx(100 / 3)
        assert mu.loc["MetalB", "CO2"] == pytest.approx(12.5)

        beta = result.process_intensities
        assert beta.loc["ExportsOfMetalB", "CO2"] == pytest.approx(14.5)

    def test_conservation_holds(self):
        model, demand = build_toy_model()
        values = {demand: 12}
        rule = ManualAllocation({"Smelting": {"MetalA": 0.8, "MetalB": 0.2}})
        result = Allocation(model, values, rule).result
        assert result.check_conservation(atol=1e-6)


class TestNoDefaultRule:
    def test_uncovered_multi_output_process_raises(self):
        model, demand = build_toy_model()
        values = {demand: 12}
        rule = ManualAllocation({})  # Smelting not covered
        with pytest.raises(ValueError, match="Smelting"):
            Allocation(model, values, rule)


class TestZeroWeightCutoffRecorded:
    def test_zero_weight_nonzero_flow_recorded_in_meta(self):
        model, demand = build_toy_model()
        values = {demand: 12}
        rule = ManualAllocation({"Smelting": {"MetalA": 1.0, "MetalB": 0.0}})
        result = Allocation(model, values, rule).result
        assert ("Smelting", "MetalB") in result.meta["cutoffs"]


class TestSupplyShares:
    def test_supply_shares_sum_to_one_per_object(self):
        model, demand = build_toy_model()
        values = {demand: 12}
        result = Allocation(model, values, MassAllocation()).result
        sigma = result.supply_shares
        for obj_id in ("Ore", "MetalA", "MetalB"):
            total = sigma[sigma["object"] == obj_id]["sigma"].sum()
            assert total == pytest.approx(1.0)


class TestZeroSupplyObject:
    def test_zero_supply_gives_nan_with_warning(self, caplog):
        import logging

        processes = [Process("P1", produces=["out"], consumes=[], exchanges=["CO2"])]
        objects = [MObject("out", has_market=True)]
        exchanges = [MExchange("CO2")]
        builder = ModelBuilder(processes, objects, exchanges)
        builder.add({builder.Y[0]: 0})
        model = builder.build({"P1": {"produces": {"out": 1.0}, "exchanges": {"CO2": 5.0}}})

        with caplog.at_level(logging.WARNING):
            result = Allocation(model, {}, MassAllocation()).result

        assert np.isnan(result.object_intensities.loc["out", "CO2"])
        assert any("zero" in rec.message.lower() or "supply" in rec.message.lower() for rec in caplog.records)

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
        result = Allocation(model, {}, MassAllocation()).result
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
        result = Allocation(model, {}, MassAllocation()).result
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
        builder.add(
            builder.pull_process_output(
                "ImportsOfraw", "raw", builder.object_production_deficit("raw")
            )
        )

        recipe = {
            "MakeMid": {"consumes": {"raw": 2.0}, "produces": {"mid": 1.0}},
            "MakeOut": {"consumes": {"mid": 3.0}, "produces": {"out": 1.0}, "exchanges": {"CO2": 1.5}},
            **fragment,
        }
        model = builder.build(recipe)

        result = Allocation(model, {demand: 1}, MassAllocation()).result
        mu_out = result.object_intensities.loc["out", "CO2"]

        # Compare to pulling a unit demand of "out" directly and summing elementary flows
        builder2 = ModelBuilder.from_structure(structure)
        builder2.add(builder2.pull_production("out", sy.S.One, until_objects=[]))
        builder2.add(
            builder2.pull_process_output(
                "ImportsOfraw", "raw", builder2.object_production_deficit("raw")
            )
        )
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
        result = Allocation(model, {demand: 100}, MassAllocation(), scope=scope).result
        assert result.check_conservation(atol=1e-6)
        # Recycled product's mu should be lower than virgin-only mu (cheaper: no Raw import burden)
        assert result.process_intensities.loc["Recycling", "CO2"] == pytest.approx(0.5)

    def test_loop_without_cutoff_solves_and_conserves(self):
        model, demand = self._model()
        scope = Scope(waste_input_burden="propagate")
        result = Allocation(model, {demand: 100}, MassAllocation(), scope=scope).result
        assert result.check_conservation(atol=1e-6)


class TestResultMeta:
    def test_meta_records_rule_and_scope(self):
        model, demand = build_toy_model()
        values = {demand: 12}
        result = Allocation(model, values, MassAllocation()).result
        assert "rule" in result.meta
        assert "scope" in result.meta
