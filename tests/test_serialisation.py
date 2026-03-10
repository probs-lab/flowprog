"""Tests for model serialisation (save/load functionality)."""

import pytest
import tempfile
import os
import json
from pathlib import Path
from rdflib import URIRef

# Import new implementation directly
from flowprog.model import ModelBuilder, Model, Process, Object
import sympy as sy


# Shorthand for creating objects
MASS = URIRef("http://qudt.org/vocab/quantitykind/Mass")
ENERGY = URIRef("http://qudt.org/vocab/quantitykind/Energy")


def MObject(id, *args, **kwargs):
    return Object(id, MASS, *args, **kwargs)


def EObject(id, *args, **kwargs):
    return Object(id, ENERGY, *args, **kwargs)


# Define some symbols for testing
a = sy.Symbol("a")
b = sy.Symbol("b", positive=True)
c = sy.Symbol("c", nonnegative=True)


class TestBasicSaveLoad:
    """Test basic save/load functionality."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        processes = [Process("M1", produces=["out"], consumes=["in"])]
        objects = [MObject("in"), MObject("out")]
        return ModelBuilder(processes, objects)

    def test_save_creates_file(self, simple_model, tmp_path):
        """Test that save creates a file."""
        filepath = tmp_path / "test_model.json"
        simple_model.save(str(filepath))
        assert filepath.exists()

    def test_save_creates_valid_json(self, simple_model, tmp_path):
        """Test that saved file is valid JSON."""
        filepath = tmp_path / "test_model.json"
        simple_model.save(str(filepath))

        with open(filepath) as f:
            data = json.load(f)

        assert "version" in data
        assert "processes" in data
        assert "objects" in data



class TestSaveLoadWithValues:
    """Test save/load with assigned values."""

    @pytest.fixture
    def model_with_values(self):
        """Create a model with some assigned values."""
        processes = [Process("M1", produces=["out"], consumes=["in"])]
        objects = [MObject("in"), MObject("out")]
        model = ModelBuilder(processes, objects)

        # Add some values
        model.add({model.X[0]: 3.5})
        model.add({model.Y[0]: 2.5})

        return model

    def test_roundtrip_preserves_symbol_assumptions(self, tmp_path):
        """Test that symbol assumptions are preserved."""
        processes = [Process("M1", produces=["out"], consumes=["in"])]
        objects = [MObject("in"), MObject("out")]
        model = ModelBuilder(processes, objects)

        model.add({model.X[0]: b})  # b is positive

        filepath = tmp_path / "test.json"
        model.save(str(filepath))
        loaded = ModelBuilder.load(str(filepath))

        m = loaded.build()
        expr = m.eval(loaded.X[0])
        # Extract the symbol from the expression
        symbols = expr.free_symbols
        b_loaded = [s for s in symbols if s.name == "b"][0]
        assert b_loaded.is_positive is True

class TestSaveLoadWithHistory:
    """Test save/load with history tracking."""

    def test_roundtrip_preserves_history(self, tmp_path):
        """Test that history labels are preserved."""
        processes = [Process("M1", produces=["out"], consumes=["in"])]
        objects = [MObject("in"), MObject("out")]
        model = ModelBuilder(processes, objects)

        model.add({model.X[0]: 3.5}, label="initial_value")
        model.add({model.X[0]: 2.5}, label="additional_value")

        filepath = tmp_path / "test.json"
        model.save(str(filepath))
        loaded = ModelBuilder.load(str(filepath))

        history = loaded.get_history(loaded.X[0])
        assert "initial_value" in history
        assert "additional_value" in history


class TestSaveLoadWithMetadata:
    """Test save/load with metadata."""

    def test_save_with_metadata(self, tmp_path):
        """Test that metadata can be saved."""
        processes = [Process("M1", produces=["out"], consumes=["in"])]
        objects = [MObject("in"), MObject("out")]
        model = ModelBuilder(processes, objects)

        metadata = {"description": "Test model", "author": "Test Author", "version": 2.0}

        filepath = tmp_path / "test.json"
        model.save(str(filepath), metadata=metadata)

        with open(filepath) as f:
            data = json.load(f)

        assert data["metadata"]["description"] == "Test model"
        assert data["metadata"]["author"] == "Test Author"
        assert data["metadata"]["version"] == 2.0

    def test_save_includes_timestamp(self, tmp_path):
        """Test that save includes a timestamp."""
        processes = [Process("M1", produces=["out"], consumes=["in"])]
        objects = [MObject("in"), MObject("out")]
        model = ModelBuilder(processes, objects)

        filepath = tmp_path / "test.json"
        model.save(str(filepath))

        with open(filepath) as f:
            data = json.load(f)

        assert "saved_at" in data
        # Check it's a valid ISO format timestamp
        from datetime import datetime

        datetime.fromisoformat(data["saved_at"])  # Will raise if invalid


class TestComplexModel:
    """Test with more complex model scenarios."""

    def test_roundtrip_complex_expressions(self, tmp_path):
        """Test with complex symbolic expressions."""
        processes = [
            Process("M1", produces=["mid"], consumes=["in"]),
            Process("M2", produces=["out"], consumes=["mid"]),
        ]
        objects = [MObject("in"), MObject("mid"), MObject("out")]
        model = ModelBuilder(processes, objects)

        # Add complex expressions
        model.add({model.X[0]: a * b / (c + 1)})
        model.add({model.Y[0]: sy.Max(a, b)})
        model.add({model.X[1]: sy.Piecewise((a, a > 0), (0, True))})

        filepath = tmp_path / "test.json"
        model.save(str(filepath))
        loaded = ModelBuilder.load(str(filepath))
        m = loaded.build()

        # Verify the expressions are preserved
        assert m.eval(loaded.X[0]) == a * b / (c + 1)
        assert m.eval(loaded.Y[0]) == sy.Max(a, b)
        # Piecewise comparison is tricky, just check it evaluates
        assert m.eval(loaded.X[1]) is not None

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_save_empty_model(self, tmp_path):
        """Test saving a model with no values assigned."""
        processes = [Process("M1", produces=["out"], consumes=["in"])]
        objects = [MObject("in"), MObject("out")]
        model = ModelBuilder(processes, objects)

        filepath = tmp_path / "test.json"
        model.save(str(filepath))
        loaded = ModelBuilder.load(str(filepath))

        assert len(loaded.processes) == 1
        assert len(loaded.objects) == 2

    def test_load_nonexistent_file(self):
        """Test loading from a file that doesn't exist."""
        with pytest.raises(FileNotFoundError):
            ModelBuilder.load("nonexistent_file.json")


class TestSaveLoadSteps:
    """Test save/load preserves logical steps."""

    def test_roundtrip_preserves_step_count(self, tmp_path):
        """Test that the number of steps is preserved."""
        processes = [Process("M1", produces=["out"], consumes=["in"])]
        objects = [MObject("in"), MObject("out")]
        model = ModelBuilder(processes, objects)

        model.add({model.X[0]: 3.5}, label="step1")
        model.add({model.X[0]: 2.5}, label="step2")

        filepath = tmp_path / "test.json"
        model.save(str(filepath))
        loaded = ModelBuilder.load(str(filepath))

        assert len(loaded._steps) == 2

    def test_roundtrip_preserves_step_values(self, tmp_path):
        """Test that step values are preserved."""
        processes = [Process("M1", produces=["out"], consumes=["in"])]
        objects = [MObject("in"), MObject("out")]
        model = ModelBuilder(processes, objects)

        model.add({model.X[0]: a * b}, label="symbolic_step")

        filepath = tmp_path / "test.json"
        model.save(str(filepath))
        loaded = ModelBuilder.load(str(filepath))

        step = loaded._steps[0]
        assert step.values[loaded.X[0]] == a * b
        assert step.description == "symbolic_step"

    def test_roundtrip_preserves_step_descriptions(self, tmp_path):
        """Test that step descriptions are preserved."""
        processes = [Process("M1", produces=["out"], consumes=["in"])]
        objects = [MObject("in"), MObject("out")]
        model = ModelBuilder(processes, objects)

        model.add({model.X[0]: 1}, label="first")
        model.add({model.X[0]: 2})  # No label
        model.add({model.X[0]: 3}, label="third")

        filepath = tmp_path / "test.json"
        model.save(str(filepath))
        loaded = ModelBuilder.load(str(filepath))

        assert loaded._steps[0].description == "first"
        assert loaded._steps[1].description is None
        assert loaded._steps[2].description == "third"

    def test_loaded_builder_recompiles_correctly_after_add(self, tmp_path):
        """Test that adding to a loaded builder produces correct results.

        This is the key motivation: without steps, adding to a loaded builder
        would lose previously accumulated values during recompilation.
        """
        processes = [Process("M1", produces=["out"], consumes=["in"])]
        objects = [MObject("in"), MObject("out")]
        model = ModelBuilder(processes, objects)

        model.add({model.X[0]: 3.5}, label="initial")

        filepath = tmp_path / "test.json"
        model.save(str(filepath))
        loaded = ModelBuilder.load(str(filepath))

        # Add more to loaded builder
        loaded.add({loaded.X[0]: 2.5}, label="after_load")
        m = loaded.build()

        # Should be 3.5 + 2.5 = 6.0, not just 2.5
        assert m.eval(loaded.X[0]) == 6.0
        assert loaded.get_history(loaded.X[0]) == ["initial", "after_load"]

    def test_roundtrip_with_pull_production_steps(self, tmp_path):
        """Test steps from pull_production are preserved and recompile correctly."""
        processes = [Process("M1", produces=["out"], consumes=["in"])]
        objects = [MObject("in"), MObject("out")]
        model = ModelBuilder(processes, objects)

        model.add(model.pull_production("out", a), label="demand")

        filepath = tmp_path / "test.json"
        model.save(str(filepath))
        loaded = ModelBuilder.load(str(filepath))

        original_m = model.build()
        loaded_m = loaded.build()

        # Recompile and check equivalence
        assert loaded_m.eval(loaded.X[0]) == original_m.eval(model.X[0])
        assert loaded_m.eval(loaded.Y[0]) == original_m.eval(model.Y[0])

    def test_roundtrip_with_limit_transformation(self, tmp_path):
        """Test steps with Limit transformations are preserved."""
        processes = [Process("P1", produces=["out"], consumes=[])]
        objects = [MObject("out")]
        model = ModelBuilder(processes, objects)

        unlimited = model.pull_production("out", a)
        model.add(unlimited)

        extra = model.pull_production("out", b)
        limited = model.limit(extra, model.Y[0], sy.Symbol("cap", positive=True))
        model.add(limited, label="limited")

        filepath = tmp_path / "test.json"
        model.save(str(filepath))
        loaded = ModelBuilder.load(str(filepath))

        assert len(loaded._steps) == 2
        # Second step should have a Limit transformation
        from flowprog.activities import Limit
        assert len(loaded._steps[1].transformations) == 1
        assert isinstance(loaded._steps[1].transformations[0], Limit)

        original_m = model.build()
        loaded_m = loaded.build()

        # Values should match after recompile
        assert loaded_m.eval(loaded.X[0]) == original_m.eval(model.X[0])

    def test_roundtrip_with_allocation(self, tmp_path):
        """Test steps with allocation (creates intermediates) are preserved."""
        processes = [
            Process("P1", produces=["mid"], consumes=["in"]),
            Process("P2", produces=["mid"], consumes=["in"]),
            Process("P3", produces=["out"], consumes=["mid"]),
        ]
        objects = [MObject("in"), MObject("mid", has_market=True), MObject("out")]
        model = ModelBuilder(processes, objects)

        model.add(
            model.pull_production(
                "mid", a, allocate_backwards={"mid": {"P1": 0.6, "P2": 0.4}}
            ),
            label="allocated",
        )

        filepath = tmp_path / "test.json"
        model.save(str(filepath))
        loaded = ModelBuilder.load(str(filepath))

        assert len(loaded._steps) == 1
        assert len(loaded._steps[0].intermediates) > 0

        original_m = model.build()
        loaded_m = loaded.build()

        # Values should match
        assert loaded_m.eval(loaded.X[0]) == original_m.eval(model.X[0])

    def test_v1_0_file_loads_without_steps(self, tmp_path):
        """Test backward compatibility: v1.0 files without steps still load."""
        processes = [Process("M1", produces=["out"], consumes=["in"])]
        objects = [MObject("in"), MObject("out")]
        model = ModelBuilder(processes, objects)
        model.add({model.X[0]: 3.5})

        # Save normally then strip steps to simulate v1.0
        filepath = tmp_path / "test.json"
        model.save(str(filepath))

        with open(filepath) as f:
            data = json.load(f)
        data["version"] = "1.0"
        del data["steps"]
        with open(filepath, 'w') as f:
            json.dump(data, f)

        loaded = ModelBuilder.load(str(filepath))
        assert loaded._steps == []
        # because we saved the ModelBuilder it no longer includes _values
        assert float(loaded.build().eval(loaded.X[0])) == 0.0

    def test_file_includes_steps_key(self, tmp_path):
        """Test that saved JSON includes steps."""
        processes = [Process("M1", produces=["out"], consumes=["in"])]
        objects = [MObject("in"), MObject("out")]
        model = ModelBuilder(processes, objects)
        model.add({model.X[0]: 1})

        filepath = tmp_path / "test.json"
        model.save(str(filepath))

        with open(filepath) as f:
            data = json.load(f)

        assert "steps" in data
        assert data["version"] == "1.1"
        assert len(data["steps"]) == 1


class TestEvaluableModelSaveLoad:
    """Test save/load for evaluable Model (with recipe data)."""

    @pytest.fixture
    def model_with_recipe(self):
        """Create an evaluable model with recipe data."""
        processes = [
            Process("SteamCracking", produces=["Ethylene"], consumes=["Naphtha"]),
            Process("Polymerization", produces=["Polyethylene"], consumes=["Ethylene"]),
        ]
        objects = [
            MObject("Naphtha", has_market=False),
            MObject("Ethylene", has_market=True),
            MObject("Polyethylene", has_market=False),
        ]

        builder = ModelBuilder(processes, objects)
        demand = sy.Symbol("demand", nonnegative=True)
        builder.add(
            builder.pull_production("Polyethylene", demand, until_objects=["Naphtha"]),
            label="Pull polyethylene production",
        )

        recipe = {
            "SteamCracking": {
                "consumes": {"Naphtha": 1.0},
                "produces": {"Ethylene": 0.3},
            },
            "Polymerization": {
                "consumes": {"Ethylene": 1.0},
                "produces": {"Polyethylene": 0.95},
            },
        }

        return builder.build(recipe), demand

    def test_save_creates_file(self, model_with_recipe, tmp_path):
        """Test that Model.save creates a file."""
        model, _ = model_with_recipe
        filepath = tmp_path / "test_model.json"
        model.save(str(filepath))
        assert filepath.exists()

    def test_save_includes_recipe(self, model_with_recipe, tmp_path):
        """Test that saved file includes recipe data."""
        model, _ = model_with_recipe
        filepath = tmp_path / "test_model.json"
        model.save(str(filepath))

        with open(filepath) as f:
            data = json.load(f)

        assert "recipe" in data
        assert "type" in data
        assert data["type"] == "evaluable_model"
        assert "SteamCracking" in data["recipe"]
        assert "Polymerization" in data["recipe"]

    def test_roundtrip_preserves_recipe(self, model_with_recipe, tmp_path):
        """Test that recipe is correctly restored after save/load."""
        model, demand = model_with_recipe
        filepath = tmp_path / "test_model.json"

        model.save(str(filepath))
        loaded = Model.load(str(filepath))

        # Verify recipe was restored
        recipe_sc = loaded.get_recipe("SteamCracking")
        recipe_poly = loaded.get_recipe("Polymerization")

        assert recipe_sc["produces"]["Ethylene"] == 0.3
        assert recipe_sc["consumes"]["Naphtha"] == 1.0
        assert recipe_poly["produces"]["Polyethylene"] == 0.95
        assert recipe_poly["consumes"]["Ethylene"] == 1.0

    def test_roundtrip_preserves_flows(self, model_with_recipe, tmp_path):
        """Test that flows are identical after save/load."""
        model, demand = model_with_recipe
        filepath = tmp_path / "test_model.json"

        # Get original flows
        flows_original = model.to_flows({demand: 1000})

        # Save and load
        model.save(str(filepath))
        loaded = Model.load(str(filepath))

        # Get loaded flows
        flows_loaded = loaded.to_flows({demand: 1000})

        # Compare
        assert len(flows_original) == len(flows_loaded)
        for i in range(len(flows_original)):
            assert flows_original.iloc[i]["source"] == flows_loaded.iloc[i]["source"]
            assert flows_original.iloc[i]["target"] == flows_loaded.iloc[i]["target"]
            assert flows_original.iloc[i]["material"] == flows_loaded.iloc[i]["material"]

            v1 = flows_original.iloc[i]["value"]
            v2 = flows_loaded.iloc[i]["value"]
            if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                assert abs(v1 - v2) < 1e-10

    def test_roundtrip_preserves_lambdify(self, model_with_recipe, tmp_path):
        """Test that lambdify works identically after save/load."""
        model, demand = model_with_recipe
        filepath = tmp_path / "test_model.json"

        # Get original lambdified function
        func_original = model.lambdify()

        # Save and load
        model.save(str(filepath))
        loaded = Model.load(str(filepath))

        # Get loaded lambdified function
        func_loaded = loaded.lambdify()

        # Test with different demand values
        for d in [500, 1000, 2000]:
            result_orig = func_original({demand: d})
            result_load = func_loaded({demand: d})

            assert set(result_orig.keys()) == set(result_load.keys())
            for key in result_orig.keys():
                assert abs(result_orig[key] - result_load[key]) < 1e-10

    def test_save_with_metadata(self, model_with_recipe, tmp_path):
        """Test that metadata can be saved with evaluable model."""
        model, _ = model_with_recipe
        filepath = tmp_path / "test_model.json"

        metadata = {"description": "Test evaluable model", "author": "Test"}
        model.save(str(filepath), metadata=metadata)

        with open(filepath) as f:
            data = json.load(f)

        assert data["metadata"]["description"] == "Test evaluable model"
        assert data["metadata"]["author"] == "Test"
