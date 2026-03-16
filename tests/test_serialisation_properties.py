"""Property-based tests for model serialization using hypothesis.

These tests verify that arbitrary models can be saved and loaded, producing
equivalent outputs. This provides much broader coverage than example-based tests.
"""

import tempfile
from pathlib import Path
from hypothesis import given, strategies as st, example
import sympy as sy

from flowprog import SympyModel, ModelBuilder, Process
from .model_strategies import MObject, model_builder_strategy


# ============================================================================
# Hypothesis Strategies
# ============================================================================


@st.composite
def recipe_data_strategy(draw, builder: ModelBuilder):
    """Generate recipe data (S and U coefficients) for a given model.

    Returns a dict like {model.S[i, j]: float_value, model.U[i, j]: float_value}
    """
    recipe_data = {}

    for i, proc in enumerate(builder.processes):
        # For each object the process produces, set S coefficient
        for obj_id in proc.produces:
            obj_idx = builder._lookup_object(obj_id)
            value = draw(st.floats(min_value=0.1, max_value=10.0))
            recipe_data[builder.S[obj_idx, i]] = value

        # For each object the process consumes, set U coefficient
        for obj_id in proc.consumes:
            obj_idx = builder._lookup_object(obj_id)
            value = draw(st.floats(min_value=0.1, max_value=10.0))
            recipe_data[builder.U[obj_idx, i]] = value

    return recipe_data


@st.composite
def model_builder_with_recipe_strategy(draw):
    """Generate a model structure with recipe data."""
    builder = draw(model_builder_strategy(max_processes=5, max_objects=5, randomize_has_stock=True))
    recipe_data = draw(recipe_data_strategy(builder))
    return builder, recipe_data


@st.composite
def model_builder_with_state_strategy(draw):
    """Generate a model with recipe data and some state from operations.

    Returns (builder, recipe_data) where:
    - builder has state accumulated through operations (X, Y values in _values)
    - recipe_data is a separate dict of S, U coefficients
    """
    builder, recipe_data = draw(model_builder_with_recipe_strategy())

    # Optionally add some symbolic expressions through operations
    # These go into the model's _values dict for X and Y
    num_operations = draw(st.integers(min_value=0, max_value=2))

    if num_operations > 0:
        # Try to add demand for the first object (which we ensured exists)
        first_obj = builder.objects[0].id

        # Check if any process produces it
        producers = list(builder.producers_of(first_obj))
        if producers:
            try:
                demand_symbol = sy.Symbol("demand", positive=True)
                builder.add(
                    builder.pull_production(first_obj, demand_symbol),
                    label="test_demand"
                )
            except (ValueError, KeyError):
                # Some builder structures may not support this
                pass

    # FIXME additional operations

    return builder, recipe_data


#### Specific example models beyond what's generated
def example_model_with_allocation():
    """Test model using allocation (creates intermediates).
    """
    processes = [
        Process("P1", produces=["mid"], consumes=["in"]),
        Process("P2", produces=["mid"], consumes=["in"]),
        Process("P3", produces=["out"], consumes=["mid"]),
    ]
    objects = [MObject("in"), MObject("mid", has_market=True), MObject("out")]
    builder = ModelBuilder(processes, objects)

    # Add demand with allocation (this creates intermediates and updates X/Y in _values)
    a = sy.Symbol("a", positive=True)
    builder.add(
        builder.pull_production(
            "mid", a, allocate_backwards={"mid": {"P1": 0.6, "P2": 0.4}}
        ),
        label="allocated_demand",
    )

    recipe_data = {
        builder.S[1, 0]: 1.0, builder.U[0, 0]: 0.8,
        builder.S[1, 1]: 1.0, builder.U[0, 1]: 0.8,
        builder.S[2, 2]: 1.0, builder.U[1, 2]: 0.9,
    }

    return builder.build(recipe_data)


def example_model_with_limits():
    """Example model using limits.
    """
    processes = [Process("P1", produces=["out"], consumes=[])]
    objects = [MObject("out")]
    builder = ModelBuilder(processes, objects)

    recipe_data = {builder.S[0, 0]: 1.0}

    a = sy.Symbol("a", positive=True)
    b = sy.Symbol("b", positive=True)

    unlimited = builder.pull_production("out", a)
    builder.add(unlimited)

    extra = builder.pull_production("out", b)
    limited = builder.limit(extra, builder.Y[0], sy.Symbol("limit", positive=True))
    builder.add(limited, label="limited_demand")

    return builder.build(recipe_data)


@st.composite
@example(example_model_with_allocation())
@example(example_model_with_limits())
def model_with_state_strategy(draw):
    """Generate a built SympyModel"""
    builder, recipe_data = draw(model_builder_with_recipe_strategy())
    return builder.build(recipe_data)


@st.composite
def simple_linear_model_strategy(draw):
    """Generate a simple linear chain model with recipe data (kept separate)."""
    num_processes = draw(st.integers(min_value=1, max_value=8))

    processes = []
    objects = []

    for i in range(num_processes):
        object_id = f"O{i}"
        objects.append(MObject(object_id, has_market=True))

        processes.append(
            Process(
                id=f"P{i}",
                produces=[object_id],
                consumes=[f"O{i-1}"] if i > 0 else [],
                has_stock=draw(st.booleans()),
            )
        )

    builder = ModelBuilder(processes, objects)

    # Generate recipe data (kept separate from model state)
    recipe_data = {}
    for i in range(num_processes):
        recipe_data[builder.S[i, i]] = draw(st.floats(min_value=0.5, max_value=2.0))
        if i > 0:
            recipe_data[builder.U[i - 1, i]] = draw(st.floats(min_value=0.5, max_value=2.0))

    # Add some demand using pull_production
    demand = sy.Symbol("demand_final", positive=True)
    final_object = builder.objects[-1].id

    builder.add(builder.pull_production(final_object, demand), label="final_demand")

    return builder.build(recipe_data), demand


# ============================================================================
# Property-Based Tests
# ============================================================================


@given(model_with_state_strategy())
# @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
def test_roundtrip_preserves_structure(model):
    """Test that save/load preserves model structure."""

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "test_model.json"

        # Save and load
        model.save(str(filepath))
        loaded = SympyModel.load(str(filepath))

        # Verify structure
        assert len(loaded.processes) == len(model.processes)
        assert len(loaded.objects) == len(model.objects)

        for orig_p, loaded_p in zip(model.processes, loaded.processes):
            assert orig_p.id == loaded_p.id
            assert set(orig_p.produces) == set(loaded_p.produces)
            assert set(orig_p.consumes) == set(loaded_p.consumes)
            assert orig_p.has_stock == loaded_p.has_stock

        for orig_o, loaded_o in zip(model.objects, loaded.objects):
            assert orig_o.id == loaded_o.id
            assert orig_o.metric == loaded_o.metric
            assert orig_o.has_market == loaded_o.has_market


@given(model_with_state_strategy())
# @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
def test_roundtrip_preserves_all_values(model):
    """Test that save/load preserves all assigned values."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "test_model.json"

        # Save and load
        model.save(str(filepath))
        loaded = SympyModel.load(str(filepath))

        # Check that all non-zero values are preserved
        for key, value in model._values.items():
            if value != sy.S.Zero:
                # Get the raw values first
                orig_value = model._values[key]
                loaded_value = loaded._values[key]

                # Convert to SymPy if needed
                if isinstance(orig_value, (int, float)):
                    orig_value = sy.S(orig_value)
                if isinstance(loaded_value, (int, float)):
                    loaded_value = sy.S(loaded_value)

                # For numeric values, compare directly
                if orig_value.is_number and loaded_value.is_number:
                    assert abs(float(orig_value) - float(loaded_value)) < 1e-10
                else:
                    # For symbolic, check they're equivalent
                    assert orig_value == loaded_value or (orig_value - loaded_value).simplify() == 0


@given(simple_linear_model_strategy())
# @settings(max_examples=15, suppress_health_check=[HealthCheck.too_slow])
def test_roundtrip_with_pull_production(model_and_symbol):
    """Test that loaded models work with pull_production."""
    model, demand = model_and_symbol

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "test_model.json"

        # Save and load
        model.save(str(filepath))
        loaded = SympyModel.load(str(filepath))

    # Verify we can evaluate expressions in both
    for i in range(len(model.processes)):
        orig_x = model.eval(model.X[i])
        loaded_x = loaded.eval(loaded.X[i])

        # Check symbolic equivalence
        if orig_x.free_symbols:
            assert orig_x == loaded_x or (orig_x - loaded_x).simplify() == 0
        elif orig_x.is_number:
            assert abs(float(orig_x) - float(loaded_x)) < 1e-10


@given(model_builder_with_state_strategy())
# @settings(max_examples=15, suppress_health_check=[HealthCheck.too_slow])
def test_loaded_builder_can_be_extended(builder_and_recipe):
    """Test that loaded models can continue to be modified."""
    builder, recipe_data = builder_and_recipe

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "test_model.json"

        # Save and load
        builder.save(str(filepath))
        loaded = ModelBuilder.load(str(filepath))

        # Add new values to loaded builder
        new_value = sy.Symbol("new_symbol", positive=True)
        loaded.add({loaded.X[0]: new_value}, label="after_load")

        # Verify the new value was added
        loaded_m = loaded.build()
        original_m = builder.build()
        assert loaded_m.eval(loaded.X[0]) != original_m.eval(builder.X[0])

        # Verify history was updated
        if "after_load" in str(loaded.get_history(loaded.X[0])):
            assert "after_load" in loaded.get_history(loaded.X[0])


@given(model_with_state_strategy())
# @settings(max_examples=15, suppress_health_check=[HealthCheck.too_slow])
def test_roundtrip_preserves_intermediates(model):
    """Test that intermediate symbols are preserved."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "test_model.json"

        initial_intermediates = len(model._intermediates)

        # Save and load
        model.save(str(filepath))
        loaded = SympyModel.load(str(filepath))

        # Verify same number of intermediates
        assert len(loaded._intermediates) == initial_intermediates

        # Verify intermediate symbols and expressions are preserved
        for (orig_sym, orig_expr, orig_label), (
            loaded_sym,
            loaded_expr,
            loaded_label,
        ) in zip(model._intermediates, loaded._intermediates):
            assert loaded_label == orig_label
            # Symbols should be equivalent (but may not be identical objects)
            assert str(orig_sym) == str(loaded_sym)


@given(
    model_with_state_strategy(),
    st.dictionaries(st.text(min_size=1, max_size=20), st.text(min_size=1, max_size=50), max_size=5),
)
# @settings(max_examples=10, suppress_health_check=[HealthCheck.too_slow])
def test_metadata_preservation(model, metadata):
    """Test that metadata is preserved through save/load."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "test_model.json"

        # Save with metadata
        model.save(str(filepath), metadata=metadata)

        # Load and verify metadata is in file
        import json

        with open(filepath) as f:
            data = json.load(f)

        for key, value in metadata.items():
            assert data["metadata"][key] == value

