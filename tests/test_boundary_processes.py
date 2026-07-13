"""Tests for boundary processes: declarative expansion of import/export/source/sink
specs into ordinary explicit processes carrying B (elementary exchange) burdens.
"""

import logging

import pytest
import sympy as sy
from rdflib import URIRef

from flowprog import ModelBuilder, ModelStructure, Process, Object, ElementaryExchange
from flowprog.boundary_processes import (
    BoundaryProcess,
    Import,
    Export,
    Source,
    Sink,
    add_boundary_processes,
)

MASS = URIRef("http://qudt.org/vocab/quantitykind/Mass")


def MObject(id, *args, **kwargs):
    return Object(id, MASS, *args, **kwargs)


def MExchange(id, *args, **kwargs):
    return ElementaryExchange(id, MASS, *args, **kwargs)


class TestConstructors:
    def test_import_is_supply_side(self):
        spec = Import("Ethylene")
        assert spec.kind == "import"
        assert spec.direction == "supply"
        assert spec.resolved_process_id() == "ImportsOfEthylene"

    def test_export_is_removal_side(self):
        spec = Export("Ethylene")
        assert spec.kind == "export"
        assert spec.direction == "removal"
        assert spec.resolved_process_id() == "ExportsOfEthylene"

    def test_source_is_supply_side(self):
        spec = Source("Electricity")
        assert spec.kind == "source"
        assert spec.direction == "supply"
        assert spec.resolved_process_id() == "SourceOfElectricity"

    def test_sink_is_removal_side(self):
        spec = Sink("Waste")
        assert spec.kind == "sink"
        assert spec.direction == "removal"
        assert spec.resolved_process_id() == "SinkOfWaste"

    def test_custom_process_id_overrides_default(self):
        spec = Import("Ethylene", process_id="EthyleneImportTerminal")
        assert spec.resolved_process_id() == "EthyleneImportTerminal"

    def test_exchanges_default_empty(self):
        spec = Import("Ethylene")
        assert spec.exchanges == {}

    def test_exchanges_can_be_passed(self):
        spec = Import("Ethylene", exchanges={"GHG_upstream_CO2e": 100})
        assert spec.exchanges == {"GHG_upstream_CO2e": 100}

    def test_invalid_kind_raises(self):
        with pytest.raises(ValueError):
            BoundaryProcess("Ethylene", "invalid_kind")


class TestAddBoundaryProcessesSupplySide:
    def _structure(self):
        processes = [Process("Cracking", produces=["Ethylene"], consumes=["Naphtha"])]
        objects = [
            MObject("Naphtha", has_market=True),
            MObject("Ethylene", has_market=True),
        ]
        exchanges = [MExchange("GHG_upstream_CO2e")]
        return ModelStructure(processes, objects, exchanges)

    def test_creates_supply_process_with_S_1(self):
        structure = self._structure()
        new_structure, fragment = add_boundary_processes(
            structure, [Import("Naphtha", exchanges={"GHG_upstream_CO2e": 423})]
        )

        proc = new_structure.processes[new_structure.lookup_process("ImportsOfNaphtha")]
        assert proc.produces == ["Naphtha"]
        assert proc.consumes == []
        assert proc.has_stock is False

        assert fragment["ImportsOfNaphtha"]["produces"] == {"Naphtha": 1}
        assert fragment["ImportsOfNaphtha"]["consumes"] == {}
        assert fragment["ImportsOfNaphtha"]["exchanges"] == {"GHG_upstream_CO2e": 423}

    def test_original_processes_preserved(self):
        structure = self._structure()
        new_structure, _ = add_boundary_processes(
            structure, [Import("Naphtha", exchanges={"GHG_upstream_CO2e": 423})]
        )
        assert "Cracking" in [p.id for p in new_structure.processes]

    def test_fragment_usable_to_build_and_evaluate(self):
        structure = self._structure()
        new_structure, fragment = add_boundary_processes(
            structure, [Import("Naphtha", exchanges={"GHG_upstream_CO2e": 423})]
        )
        builder = ModelBuilder.from_structure(new_structure)
        builder.add(
            builder.pull_production("Ethylene", sy.Symbol("demand", positive=True))
        )
        recipe = {
            "Cracking": {"consumes": {"Naphtha": 3.0}, "produces": {"Ethylene": 1.0}},
            **fragment,
        }
        model = builder.build(recipe)

        total_co2e = model.eval(model.structure.ElementaryBalance[0]).subs(
            {sy.Symbol("demand", positive=True): 10}
        )
        assert float(total_co2e) == pytest.approx(10 * 3.0 * 423)


class TestAddBoundaryProcessesRemovalSide:
    def _structure(self):
        processes = [Process("Use", produces=["Waste"], consumes=["Product"])]
        objects = [
            MObject("Product", has_market=True),
            MObject("Waste", has_market=True),
        ]
        exchanges = [MExchange("CH4_landfill")]
        return ModelStructure(processes, objects, exchanges)

    def test_creates_removal_process_with_U_1(self):
        structure = self._structure()
        new_structure, fragment = add_boundary_processes(
            structure, [Sink("Waste", exchanges={"CH4_landfill": 0.05})]
        )

        proc = new_structure.processes[new_structure.lookup_process("SinkOfWaste")]
        assert proc.produces == []
        assert proc.consumes == ["Waste"]
        assert proc.has_stock is False

        assert fragment["SinkOfWaste"]["consumes"] == {"Waste": 1}
        assert fragment["SinkOfWaste"]["produces"] == {}
        assert fragment["SinkOfWaste"]["exchanges"] == {"CH4_landfill": 0.05}


class TestValidation:
    def _structure(self):
        processes = [Process("P1", produces=["out"], consumes=[])]
        objects = [MObject("out", has_market=True)]
        exchanges = [MExchange("CO2")]
        return ModelStructure(processes, objects, exchanges)

    def test_unknown_object_raises(self):
        structure = self._structure()
        with pytest.raises(ValueError):
            add_boundary_processes(structure, [Import("DoesNotExist")])

    def test_unknown_exchange_raises(self):
        structure = self._structure()
        with pytest.raises(ValueError):
            add_boundary_processes(
                structure, [Import("out", exchanges={"N2O": 1.0})]
            )

    def test_process_id_collision_with_existing_process_raises(self):
        structure = self._structure()
        with pytest.raises(ValueError):
            add_boundary_processes(structure, [Import("out", process_id="P1")])

    def test_process_id_collision_within_specs_raises(self):
        structure = self._structure()
        with pytest.raises(ValueError):
            add_boundary_processes(
                structure, [Import("out"), Import("out")]
            )

    def test_warns_if_object_has_no_market(self, caplog):
        processes = [Process("P1", produces=["out"], consumes=[])]
        objects = [MObject("out", has_market=False)]
        structure = ModelStructure(processes, objects, [MExchange("CO2")])

        with caplog.at_level(logging.WARNING):
            add_boundary_processes(structure, [Export("out")])

        assert any("has_market" in rec.message or "market" in rec.message for rec in caplog.records)

    def test_no_warning_if_object_has_market(self, caplog):
        structure = self._structure()
        with caplog.at_level(logging.WARNING):
            add_boundary_processes(structure, [Export("out")])
        assert len(caplog.records) == 0


class TestExpansionMetadata:
    def test_records_spec_metadata_for_reporting(self):
        processes = [Process("P1", produces=["out"], consumes=[])]
        objects = [MObject("out", has_market=True)]
        structure = ModelStructure(processes, objects, [MExchange("CO2")])

        new_structure, _ = add_boundary_processes(structure, [Import("out")])

        assert "ImportsOfout" in new_structure.boundary_process_specs
        spec = new_structure.boundary_process_specs["ImportsOfout"]
        assert spec.kind == "import"
        assert spec.object_id == "out"


class TestRemovalSideExpansion:
    """An Export/Sink spec generates a memoryless process; dispatching
    ConsumptionDeficit through it balances the market; its Y-keyed B entries
    evaluate to burden x throughput (not zero)."""

    def test_sink_dispatch_balances_market_and_B_entries_nonzero(self):
        processes = [Process("Produce", produces=["Waste"], consumes=[])]
        objects = [MObject("Waste", has_market=True)]
        exchanges = [MExchange("CH4_landfill")]
        structure = ModelStructure(processes, objects, exchanges)

        new_structure, fragment = add_boundary_processes(
            structure, [Sink("Waste", exchanges={"CH4_landfill": 0.05})]
        )
        builder = ModelBuilder.from_structure(new_structure)

        demand = sy.Symbol("demand", positive=True)
        builder.add(builder.pull_production("Waste", demand))
        builder.add(
            builder.push_process_input(
                "SinkOfWaste", "Waste", builder.object_consumption_deficit("Waste")
            )
        )

        recipe = {
            "Produce": {"produces": {"Waste": 1.0}},
            **fragment,
        }
        model = builder.build(recipe)

        # Market balances: nothing left over
        balance = model.eval(builder.object_balance("Waste")).subs({demand: 10})
        assert float(balance) == pytest.approx(0, abs=1e-9)

        # Sink is memoryless: X == Y == throughput
        j = new_structure.lookup_process("SinkOfWaste")
        y_sink = model.eval(model.Y[j]).subs({demand: 10})
        x_sink = model.eval(model.X[j]).subs({demand: 10})
        assert float(y_sink) == pytest.approx(10)
        assert float(x_sink) == pytest.approx(10)

        # Y-keyed B entry evaluates to burden * throughput, not zero
        co2 = model.eval(model.structure.ElementaryBalance[0]).subs({demand: 10})
        assert float(co2) == pytest.approx(0.05 * 10)


class TestImportsWithUpstreamEmissions:
    def test_supply_side_pull_remaining_deficit(self):
        processes = [
            Process("Domestic", produces=["Ethylene"], consumes=[]),
            Process("Use", produces=[], consumes=["Ethylene"]),
        ]
        objects = [MObject("Ethylene", has_market=True)]
        exchanges = [MExchange("GHG_upstream_CO2e")]
        structure = ModelStructure(processes, objects, exchanges)

        new_structure, fragment = add_boundary_processes(
            structure, [Import("Ethylene", exchanges={"GHG_upstream_CO2e": 500})]
        )
        builder = ModelBuilder.from_structure(new_structure)

        demand = sy.Symbol("demand", positive=True)
        capacity = sy.Symbol("capacity", nonnegative=True)

        # Downstream use creates consumption (demand) for Ethylene
        builder.add(builder.push_consumption("Ethylene", demand), label="use demand")

        # Domestic production, capped
        proposal = builder.pull_process_output("Domestic", "Ethylene", demand)
        limited = builder.limit(proposal, builder.Y[builder._lookup_process("Domestic")], capacity)
        builder.add(limited, label="domestic production, capped")

        # Import makes up the remainder
        builder.add(
            builder.pull_process_output(
                "ImportsOfEthylene",
                "Ethylene",
                builder.object_production_deficit("Ethylene"),
            ),
            label="import remainder",
        )

        recipe = {
            "Domestic": {"produces": {"Ethylene": 1.0}},
            "Use": {"consumes": {"Ethylene": 1.0}},
            **fragment,
        }
        model = builder.build(recipe)

        data = {demand: 100, capacity: 30}
        domestic = model.eval(model.Y[builder._lookup_process("Domestic")]).subs(data)
        imported = model.eval(
            model.Y[new_structure.lookup_process("ImportsOfEthylene")]
        ).subs(data)

        assert float(domestic) == pytest.approx(30)
        assert float(imported) == pytest.approx(70)
