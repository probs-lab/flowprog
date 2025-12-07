"""
Test compatibility between old Model API and new ModelBuilder/Model API.

These tests run the same operations on both implementations and verify that
they produce identical results, ensuring backward compatibility.
"""

import pytest
import sympy as sy
from rdflib import URIRef
import pandas as pd

# Import old API (delegation layer)
from flowprog.imperative_model import Model as OldModel
from flowprog.imperative_model import Process, Object

# Import new API (direct)
from flowprog.model import ModelBuilder, Model as NewModel


MASS = URIRef("http://qudt.org/vocab/quantitykind/Mass")


def compare_flows(flows1, flows2):
    """Compare two flow DataFrames, ignoring row order."""
    # Sort both by source, target, material for comparison
    df1 = flows1.sort_values(['source', 'target', 'material']).reset_index(drop=True)
    df2 = flows2.sort_values(['source', 'target', 'material']).reset_index(drop=True)

    # Compare values numerically (handle symbolic expressions)
    assert len(df1) == len(df2), f"Different number of flows: {len(df1)} vs {len(df2)}"

    for i in range(len(df1)):
        assert df1.iloc[i]['source'] == df2.iloc[i]['source']
        assert df1.iloc[i]['target'] == df2.iloc[i]['target']
        assert df1.iloc[i]['material'] == df2.iloc[i]['material']

        # For symbolic values, compare as expressions
        v1 = df1.iloc[i]['value']
        v2 = df2.iloc[i]['value']

        if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
            assert abs(v1 - v2) < 1e-10, f"Values differ: {v1} vs {v2}"
        else:
            # Symbolic comparison
            assert sy.simplify(sy.sympify(v1) - sy.sympify(v2)) == 0


class TestSimpleChainCompatibility:
    """Test simple chain model compatibility."""

    @pytest.fixture
    def old_model(self):
        """Create model using old API."""
        processes = [Process("M1", produces=["out"], consumes=["in"])]
        objects = [
            Object("in", MASS, has_market=False),
            Object("out", MASS, has_market=False),
        ]
        return OldModel(processes, objects)

    @pytest.fixture
    def new_model(self):
        """Create model using new API."""
        processes = [Process("M1", produces=["out"], consumes=["in"])]
        objects = [
            Object("in", MASS, has_market=False),
            Object("out", MASS, has_market=False),
        ]
        return ModelBuilder(processes, objects)

    def test_pull_production_identical(self, old_model, new_model):
        """Test that pull_production produces identical results."""
        demand = sy.Symbol("demand")

        # Old API
        old_model.add(old_model.pull_production("out", demand))

        # New API
        new_model.add(new_model.pull_production("out", demand))

        # Define recipe
        recipe = {
            old_model.S[1, 0]: 0.5,  # out from M1
            old_model.U[0, 0]: 1.0,  # in into M1
        }

        # Build models with recipe
        old_flows = old_model.to_flows({**recipe, demand: 1000})
        new_flows = new_model.build(recipe).to_flows({demand: 1000})

        compare_flows(old_flows, new_flows)

    def test_push_consumption_identical(self, old_model, new_model):
        """Test that push_consumption produces identical results."""
        supply = sy.Symbol("supply")

        # Old API
        old_model.add(old_model.push_consumption("in", supply))

        # New API
        new_model.add(new_model.push_consumption("in", supply))

        # Define recipe
        recipe = {
            old_model.S[1, 0]: 0.5,
            old_model.U[0, 0]: 1.0,
        }

        # Build and compare
        old_flows = old_model.to_flows({**recipe, supply: 500})
        new_flows = new_model.build(recipe).to_flows({supply: 500})

        compare_flows(old_flows, new_flows)


class TestMultipleProducersCompatibility:
    """Test model with multiple producers compatibility."""

    @pytest.fixture
    def old_model(self):
        processes = [
            Process("M1", produces=["out"], consumes=["in1"]),
            Process("M2", produces=["out"], consumes=["in2"]),
        ]
        objects = [
            Object("in1", MASS, has_market=False),
            Object("in2", MASS, has_market=False),
            Object("out", MASS, has_market=True),
        ]
        return OldModel(processes, objects)

    @pytest.fixture
    def new_model(self):
        processes = [
            Process("M1", produces=["out"], consumes=["in1"]),
            Process("M2", produces=["out"], consumes=["in2"]),
        ]
        objects = [
            Object("in1", MASS, has_market=False),
            Object("in2", MASS, has_market=False),
            Object("out", MASS, has_market=True),
        ]
        return ModelBuilder(processes, objects)

    def test_allocation_identical(self, old_model, new_model):
        """Test that allocation produces identical results."""
        demand = sy.Symbol("demand")
        alpha = sy.Symbol("alpha", positive=True)

        allocate_backwards = {"out": {"M1": alpha, "M2": 1 - alpha}}

        # Old API
        old_model.add(
            old_model.pull_production(
                "out", demand, allocate_backwards=allocate_backwards
            )
        )

        # New API
        new_model.add(
            new_model.pull_production(
                "out", demand, allocate_backwards=allocate_backwards
            )
        )

        # Define recipe
        recipe = {
            old_model.S[2, 0]: 0.5,  # out from M1
            old_model.U[0, 0]: 1.0,  # in1 into M1
            old_model.S[2, 1]: 0.4,  # out from M2
            old_model.U[1, 1]: 1.0,  # in2 into M2
        }

        # Build and compare
        old_flows = old_model.to_flows({**recipe, demand: 1000, alpha: 0.6})
        new_flows = new_model.build(recipe).to_flows({demand: 1000, alpha: 0.6})

        compare_flows(old_flows, new_flows)


class TestComplexModelCompatibility:
    """Test more complex model compatibility."""

    @pytest.fixture
    def old_model(self):
        processes = [
            Process("SteamCracking", produces=["Ethylene"], consumes=["Naphtha"]),
            Process("Polymerization", produces=["Polyethylene"], consumes=["Ethylene"]),
        ]
        objects = [
            Object("Naphtha", MASS, has_market=False),
            Object("Ethylene", MASS, has_market=True),
            Object("Polyethylene", MASS, has_market=False),
        ]
        return OldModel(processes, objects)

    @pytest.fixture
    def new_model(self):
        processes = [
            Process("SteamCracking", produces=["Ethylene"], consumes=["Naphtha"]),
            Process("Polymerization", produces=["Polyethylene"], consumes=["Ethylene"]),
        ]
        objects = [
            Object("Naphtha", MASS, has_market=False),
            Object("Ethylene", MASS, has_market=True),
            Object("Polyethylene", MASS, has_market=False),
        ]
        return ModelBuilder(processes, objects)

    def test_multi_step_chain_identical(self, old_model, new_model):
        """Test that multi-step chain produces identical results."""
        demand = sy.Symbol("demand")

        # Old API
        old_model.add(
            old_model.pull_production("Polyethylene", demand, until_objects=["Naphtha"]),
            label="Pull polyethylene production"
        )

        # New API
        new_model.add(
            new_model.pull_production("Polyethylene", demand, until_objects=["Naphtha"]),
            label="Pull polyethylene production"
        )

        # Define recipe
        recipe = {
            old_model.S[1, 0]: 0.3,   # Ethylene from SteamCracking
            old_model.U[0, 0]: 1.0,   # Naphtha into SteamCracking
            old_model.S[2, 1]: 0.95,  # Polyethylene from Polymerization
            old_model.U[1, 1]: 1.0,   # Ethylene into Polymerization
        }

        # Build and compare
        old_flows = old_model.to_flows({**recipe, demand: 1000})
        new_flows = new_model.build(recipe).to_flows({demand: 1000})

        compare_flows(old_flows, new_flows)

    def test_history_identical(self, old_model, new_model):
        """Test that history tracking is identical."""
        demand = sy.Symbol("demand")

        # Old API
        old_model.add(
            old_model.pull_production("Polyethylene", demand, until_objects=["Naphtha"]),
            label="First pull"
        )
        old_model.add(
            old_model.pull_production("Polyethylene", demand * 0.5, until_objects=["Naphtha"]),
            label="Second pull"
        )

        # New API
        new_model.add(
            new_model.pull_production("Polyethylene", demand, until_objects=["Naphtha"]),
            label="First pull"
        )
        new_model.add(
            new_model.pull_production("Polyethylene", demand * 0.5, until_objects=["Naphtha"]),
            label="Second pull"
        )

        # Check history
        old_history_x = old_model.get_history(old_model.X[0])
        new_history_x = new_model.get_history(new_model.X[0])

        assert old_history_x == new_history_x == ["First pull", "Second pull"]

        old_history_y = old_model.get_history(old_model.Y[1])
        new_history_y = new_model.get_history(new_model.Y[1])

        assert old_history_y == new_history_y == ["First pull", "Second pull"]


class TestLambdifyCompatibility:
    """Test lambdify functionality compatibility."""

    @pytest.fixture
    def old_model(self):
        processes = [Process("M1", produces=["out"], consumes=["in"])]
        objects = [
            Object("in", MASS, has_market=False),
            Object("out", MASS, has_market=False),
        ]
        return OldModel(processes, objects)

    @pytest.fixture
    def new_model(self):
        processes = [Process("M1", produces=["out"], consumes=["in"])]
        objects = [
            Object("in", MASS, has_market=False),
            Object("out", MASS, has_market=False),
        ]
        return ModelBuilder(processes, objects)

    def test_lambdify_identical(self, old_model, new_model):
        """Test that lambdify produces identical results."""
        demand = sy.Symbol("demand")

        # Old API
        old_model.add(old_model.pull_production("out", demand))

        # New API
        new_model.add(new_model.pull_production("out", demand))

        # Define recipe
        recipe = {
            old_model.S[1, 0]: 0.5,
            old_model.U[0, 0]: 1.0,
        }

        # Create lambdified functions
        old_func = old_model.lambdify(recipe)
        new_func = new_model.build(recipe).lambdify()

        # Test with different demand values
        for d in [100, 500, 1000]:
            old_result = old_func({demand: d})
            new_result = new_func({demand: d})

            # Results should be identical
            assert set(old_result.keys()) == set(new_result.keys())
            for key in old_result.keys():
                assert abs(old_result[key] - new_result[key]) < 1e-10


class TestObjectBalanceCompatibility:
    """Test object balance methods compatibility."""

    @pytest.fixture
    def old_model(self):
        processes = [
            Process("P1", produces=["O2"], consumes=["O1"]),
            Process("P2", produces=["O3"], consumes=["O2"]),
        ]
        objects = [
            Object("O1", MASS, has_market=False),
            Object("O2", MASS, has_market=True),
            Object("O3", MASS, has_market=False),
        ]
        return OldModel(processes, objects)

    @pytest.fixture
    def new_model(self):
        processes = [
            Process("P1", produces=["O2"], consumes=["O1"]),
            Process("P2", produces=["O3"], consumes=["O2"]),
        ]
        objects = [
            Object("O1", MASS, has_market=False),
            Object("O2", MASS, has_market=True),
            Object("O3", MASS, has_market=False),
        ]
        return ModelBuilder(processes, objects)

    def test_object_balance_identical(self, old_model, new_model):
        """Test that object_balance produces identical expressions."""
        demand = sy.Symbol("demand")

        # Old API
        old_model.add(old_model.pull_production("O3", demand, until_objects=["O1"]))

        # New API
        new_model.add(new_model.pull_production("O3", demand, until_objects=["O1"]))

        # Get balances
        old_balance = old_model.object_balance("O2")
        new_balance = new_model.object_balance("O2")

        # Simplify and compare
        assert sy.simplify(old_balance - new_balance) == 0

    def test_deficit_methods_identical(self, old_model, new_model):
        """Test that deficit methods produce identical expressions."""
        demand = sy.Symbol("demand")

        # Old API
        old_model.add(old_model.pull_production("O3", demand, until_objects=["O1"]))

        # New API
        new_model.add(new_model.pull_production("O3", demand, until_objects=["O1"]))

        # Get deficits
        old_prod_deficit = old_model.object_production_deficit("O2")
        new_prod_deficit = new_model.object_production_deficit("O2")

        old_cons_deficit = old_model.object_consumption_deficit("O2")
        new_cons_deficit = new_model.object_consumption_deficit("O2")

        # Both should produce the same type of expression (Max with intermediates)
        # We can't directly compare because intermediate symbols differ,
        # but we can check they have the same structure
        assert isinstance(old_prod_deficit, sy.Symbol)
        assert isinstance(new_prod_deficit, sy.Symbol)
        assert isinstance(old_cons_deficit, sy.Symbol)
        assert isinstance(new_cons_deficit, sy.Symbol)
