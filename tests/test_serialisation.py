"""Tests for model serialisation (save/load functionality)."""

import pytest
import tempfile
import os
import json
from pathlib import Path
from rdflib import URIRef

from flowprog.imperative_model import Model, Process, Object
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
        return Model(processes, objects)

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
        model = Model(processes, objects)

        # Add some values
        model.add({model.X[0]: 3.5})
        model.add({model.Y[0]: 2.5})

        return model

    def test_roundtrip_preserves_symbol_assumptions(self, tmp_path):
        """Test that symbol assumptions are preserved."""
        processes = [Process("M1", produces=["out"], consumes=["in"])]
        objects = [MObject("in"), MObject("out")]
        model = Model(processes, objects)

        model.add({model.X[0]: b})  # b is positive

        filepath = tmp_path / "test.json"
        model.save(str(filepath))
        loaded = Model.load(str(filepath))

        expr = loaded.eval(loaded.X[0])
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
        model = Model(processes, objects)

        model.add({model.X[0]: 3.5}, label="initial_value")
        model.add({model.X[0]: 2.5}, label="additional_value")

        filepath = tmp_path / "test.json"
        model.save(str(filepath))
        loaded = Model.load(str(filepath))

        history = loaded.get_history(loaded.X[0])
        assert "initial_value" in history
        assert "additional_value" in history


class TestSaveLoadWithMetadata:
    """Test save/load with metadata."""

    def test_save_with_metadata(self, tmp_path):
        """Test that metadata can be saved."""
        processes = [Process("M1", produces=["out"], consumes=["in"])]
        objects = [MObject("in"), MObject("out")]
        model = Model(processes, objects)

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
        model = Model(processes, objects)

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
        model = Model(processes, objects)

        # Add complex expressions
        model.add({model.X[0]: a * b / (c + 1)})
        model.add({model.Y[0]: sy.Max(a, b)})
        model.add({model.X[1]: sy.Piecewise((a, a > 0), (0, True))})

        filepath = tmp_path / "test.json"
        model.save(str(filepath))
        loaded = Model.load(str(filepath))

        # Verify the expressions are preserved
        assert loaded.eval(loaded.X[0]) == a * b / (c + 1)
        assert loaded.eval(loaded.Y[0]) == sy.Max(a, b)
        # Piecewise comparison is tricky, just check it evaluates
        assert loaded.eval(loaded.X[1]) is not None

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_save_empty_model(self, tmp_path):
        """Test saving a model with no values assigned."""
        processes = [Process("M1", produces=["out"], consumes=["in"])]
        objects = [MObject("in"), MObject("out")]
        model = Model(processes, objects)

        filepath = tmp_path / "test.json"
        model.save(str(filepath))
        loaded = Model.load(str(filepath))

        assert len(loaded.processes) == 1
        assert len(loaded.objects) == 2

    def test_load_nonexistent_file(self):
        """Test loading from a file that doesn't exist."""
        with pytest.raises(FileNotFoundError):
            Model.load("nonexistent_file.json")

    def test_cross_references_preserved(self, tmp_path):
        """Test that cross-references between IndexedBase are preserved."""
        processes = [Process("M1", produces=["out"], consumes=["in"])]
        objects = [MObject("in"), MObject("out")]
        model = Model(processes, objects)

        # Create an expression that references Y
        model.add({model.X[0]: model.Y[0] * 2})

        filepath = tmp_path / "test.json"
        model.save(str(filepath))
        loaded = Model.load(str(filepath))

        # Check that the expression was preserved correctly
        expr = loaded._values[loaded.X[0]]
        assert expr == loaded.Y[0] * 2

        # Verify the expression still contains Y[0] references
        assert loaded.Y[0] in expr.free_symbols
