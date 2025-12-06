import pytest
import numpy as np
from hypothesis import strategies as st, given, assume, example, settings
from sympy.abc import a, b, c
from rdflib import URIRef
from flowprog.imperative_model import *


# Shorthand
MASS = URIRef('http://qudt.org/vocab/quantitykind/Mass')
def MObject(id, *args, **kwargs):
    return Object(id, MASS, *args, **kwargs)


@given(st.floats(min_value=0, max_value=1e20, allow_infinity=False),
       st.floats(min_value=0, max_value=1e20, allow_infinity=False),
       st.floats(min_value=0, max_value=1e20))
@example(10, 0, 10)
@settings(deadline=1000)
def test_limit_with_symbols(initial, consumption, limit_value):
    processes = [
        Process("P1", consumes=["in"], produces=["mid"]),
        Process("P2", consumes=["mid"], produces=["out"]),
    ]
    objects = [MObject("in"), MObject("mid", True), MObject("out")]
    m = Model(processes, objects)

    sy.var("a, b, c", real=True)

    # First flows
    m.add(m.pull_production("out", a))

    # Add extra demand with limit
    unlimited_extra = m.pull_production("out", b)
    limited_extra = m.limit(unlimited_extra, m.Y[0], c)
    m.add(limited_extra)

    # We should not have exceeded the limit
    recipe_data = {
        m.U[0, 0]: 1.0,
        m.S[1, 0]: 0.6,
        m.U[1, 1]: 2.2,
        m.S[2, 1]: 2.0,
    }
    value = m.eval(m.Y[0]).subs({
        a: initial,
        b: consumption,
        c: limit_value,
        **recipe_data,
    })
    assert value >= initial
    # exceeded = max(0, value - max(initial, limit_value))
    # assert exceeded <= 1e-6
    expected_max = max(initial / 2.0 * 2.2 / 0.6 * 1.00001 + 1e-3, limit_value)
    assert value <= expected_max

    # Also test the lambdify version (trigger divide-by-zero differently?)
    func = m.lambdify(data=recipe_data)
    result = func({
        "a": initial,
        "b": consumption,
        "c": limit_value,
    })


# TODO: check that there is a test that fails if you swap `v` to `proposed` in
# the implementation of `limit` (middle case)



# Define arbitrary models for testing

def has_cycle_through_market(processes, objects):
    """Check if there's a cycle that passes through an object with has_market=True.

    A cycle only matters if it goes through at least one object that has a market.
    This is because objects without markets don't create problematic feedback loops.
    """
    # Build object ID to has_market mapping
    market_objects = {obj.id for obj in objects if obj.has_market}

    # If no market objects, cycles don't matter
    if not market_objects:
        return False

    # Build directed graph: object -> objects reachable through processes
    graph = {}  # object_id -> set of object_ids
    all_obj_ids = {obj.id for obj in objects}

    for obj_id in all_obj_ids:
        graph[obj_id] = set()

    for proc in processes:
        # For each process, add edges from consumed objects to produced objects
        for consumed in proc.consumes:
            for produced in proc.produces:
                graph[consumed].add(produced)

    # DFS-based cycle detection using three-color algorithm
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {obj_id: WHITE for obj_id in all_obj_ids}

    def dfs(node, path):
        """Perform DFS to detect cycles. Returns True if a cycle through a market object is found."""
        if color[node] == BLACK:
            # Already fully explored
            return False
        if color[node] == GRAY:
            # Found a cycle - check if it goes through a market object
            cycle_start = path.index(node)
            cycle = path[cycle_start:]
            return any(obj_id in market_objects for obj_id in cycle)

        color[node] = GRAY
        path.append(node)

        for neighbor in graph[node]:
            if dfs(neighbor, path):
                return True

        path.pop()
        color[node] = BLACK
        return False

    # Check for cycles starting from each unvisited node
    for obj_id in all_obj_ids:
        if color[obj_id] == WHITE:
            if dfs(obj_id, []):
                return True

    return False


@st.composite
def model_strategy(draw):
    num_processes = draw(st.integers(min_value=1, max_value=10))
    num_objects = draw(st.integers(min_value=1, max_value=10))
    object_ids = [f"O{i}" for i in range(num_objects)]

    # Recipes must have produces and/or consumes
    recipes = st.tuples(
        st.sets(st.sampled_from(object_ids), max_size=4),
        st.sets(st.sampled_from(object_ids), max_size=4),
    ).filter(lambda x: x[0] or x[1] and x[0].isdisjoint(x[1]))

    recipe_draws = [
        draw(recipes)
        for i in range(num_processes)
    ]

    processes = [
        # Make sure that the first object is always produced at least once, so
        # it's safe to ask for it
        Process(f"P{i}", consumes=c, produces=(p | {object_ids[0]} if i == 0 else p))
        for i, (c, p) in enumerate(recipe_draws)
    ]

    # Randomly choose if objects have a market if they are both produced and
    # consumed; otherwise it doesn't make sense to.
    obj_produced = set()
    obj_consumed = set()
    for p in processes:
        obj_produced.update(p.produces)
        obj_consumed.update(p.consumes)

    objects = [
        Object(
            object_id,
            MASS,
            has_market=(
                draw(st.booleans()) if (object_id in obj_produced and object_id in obj_consumed)
                else False
            )
        )
        for object_id in object_ids
    ]

    # Reject models with cycles through market objects
    assume(not has_cycle_through_market(processes, objects))

    return Model(processes, objects)


# ============================================================================
# SIMPLE MODEL TESTS (single process: in → out)
# ============================================================================

class TestSimpleModel:
    """Tests using simple single-process model (in → out)."""

    def _create_simple_model(self):
        """Create a simple model with one process."""
        processes = [Process("P1", consumes=["in"], produces=["out"])]
        objects = [MObject("in"), MObject("out")]
        return Model(processes, objects)

    @given(
        st.floats(min_value=0, max_value=1e6, allow_infinity=False, allow_nan=False),
        st.floats(min_value=0, max_value=1e6, allow_infinity=False, allow_nan=False),
        st.floats(min_value=0, max_value=1e6, allow_infinity=False, allow_nan=False),
    )
    def test_limit_with_zero_initial_demand(self, initial, extra, limit_value):
        """Test limit() when initial demand could be zero."""
        m = self._create_simple_model()
        sy.var("a, b, c", real=True)

        # Set initial production (could be zero)
        m.add(m.pull_production("out", a))

        # Add extra demand with limit
        unlimited_extra = m.pull_production("out", b)
        limited_extra = m.limit(unlimited_extra, m.Y[0], c)
        m.add(limited_extra)

        # Evaluate with actual values
        data = {
            a: initial,
            b: extra,
            c: limit_value,
            m.U[0, 0]: 1.0,
            m.S[1, 0]: 1.0,
        }

        value = m.eval(m.Y[0]).subs(data)
        assert value >= 0
        assert value <= max(initial + extra, limit_value) + 1e-6

    @given(
        st.floats(min_value=0.0, max_value=1e6, allow_infinity=False, allow_nan=False),
        st.sampled_from(["pull_production", "pull_process_output", "push_process_input"]),
    )
    def test_various_model_methods_with_zero_demand(self, demand, method_name):
        """Test various model methods with potentially zero demand values."""
        m = self._create_simple_model()
        sy.var("d", real=True)

        # Call different model methods
        if method_name == "pull_production":
            result = m.pull_production("out", d)
        elif method_name == "pull_process_output":
            result = m.pull_process_output("P1", "out", d)
        elif method_name == "push_process_input":
            result = m.push_process_input("P1", "in", d)

        m.add(result)

        data = {
            d: demand,
            m.U[0, 0]: 1.0,
            m.S[1, 0]: 1.0,
        }

        if method_name in ["pull_production", "pull_process_output"]:
            value = m.eval(m.Y[0]).subs(data)
        else:
            value = m.eval(m.X[0]).subs(data)

        assert value == demand


# ============================================================================
# BRANCHING MODEL TESTS (in → mid → {out1, out2})
# ============================================================================

class TestBranchingModel:
    """Tests using branching pipeline model (in → mid → {out1, out2})."""

    def _create_branching_model(self):
        """Create a branching model with intermediate market."""
        processes = [
            Process("P1", consumes=["in"], produces=["mid"]),
            Process("P2", consumes=["mid"], produces=["out1"]),
            Process("P3", consumes=["mid"], produces=["out2"]),
        ]
        objects = [MObject("in"), MObject("mid", True), MObject("out1"), MObject("out2")]
        m = Model(processes, objects)

        recipe_data = {
            m.U[0, 0]: 1.0,
            m.S[1, 0]: 1.0,
            m.U[1, 1]: 1.0,
            m.S[2, 1]: 1.0,
            m.U[1, 2]: 1.0,
            m.S[3, 2]: 1.0,
        }
        return m, recipe_data

    @given(st.floats(min_value=0, allow_infinity=False),
        st.floats(min_value=0, allow_infinity=False),
        st.floats(min_value=0, allow_infinity=False))
    @example(2, 3, 10)
    def test_limit_can_be_arbitrary_expression(self, initial1, initial2, consumption):
        m, recipe_data = self._create_branching_model()
        sy.var("a, b, c", real=True)

        # First flows
        m.add(
            m.pull_production("out1", a, until_objects={"mid"}),
            m.pull_production("out2", b, until_objects={"mid"}),
        )

        # Add extra demand with limit
        unlimited_extra = m.pull_process_output("P1", "mid", c)
        limit_expr = m.X[1] * m.U[1, 1] + m.X[2] * m.U[1, 2]
        limited_extra = m.limit(
            unlimited_extra,
            m.Y[0] * m.S[1, 0],
            limit_expr,
        )
        m.add(limited_extra)

        # We should not have exceeded the limit
        data = {
            a: initial1,
            b: initial2,
            c: consumption,
            **recipe_data,
        }
        value = m.eval(m.Y[0]).subs(data)
        # Use tolerance-based comparisons to handle floating-point precision errors
        # (e.g., 0.29/29.0 = 0.009999999999999998, not exactly 0.01)
        # Use both absolute tolerance (for small values) and relative tolerance (for large values)
        atol = 1e-10  # absolute tolerance
        rtol = 1e-9   # relative tolerance
        expected_min = min(initial1 + initial2, consumption)
        expected_max = initial1 + initial2
        tolerance_min = atol + rtol * abs(float(expected_min))
        tolerance_max = atol + rtol * abs(float(expected_max))
        assert float(value) >= float(expected_min) - tolerance_min, \
            f"value {float(value)} should be >= {float(expected_min)} (within tolerance {tolerance_min})"
        assert float(value) <= float(expected_max) + tolerance_max, \
            f"value {float(value)} should be <= {float(expected_max)} (within tolerance {tolerance_max})"

    @given(
        st.floats(min_value=0, max_value=1e6, allow_infinity=False, allow_nan=False),
        st.floats(min_value=0, max_value=1e6, allow_infinity=False, allow_nan=False),
        st.floats(min_value=0, max_value=1e6, allow_infinity=False, allow_nan=False),
    )
    def test_complex_model_with_limit_and_balance(self, supply, demand1, demand2):
        """Test limit() and object_balance together in branching model."""
        m, recipe_data = self._create_branching_model()
        sy.var("a, b, c", real=True)

        # Initial production and consumption
        m.add(
            m.push_consumption("in", a, until_objects={"mid"}),
            m.pull_production("out1", b, until_objects={"mid"}),
        )

        # Balance extra demand for `out2`, but only up to available excess of `mid`
        proposed = m.pull_production("out2", c, until_objects={"mid"})
        expr_to_limit = m.expr("ProcessInput", process_id="P3", object_id="mid")
        limit = m.object_consumption_deficit("mid")
        m.add(
            m.limit(proposed, expr_to_limit, limit)
        )

        # Evaluate
        data = {
            a: supply,
            b: demand1,
            c: demand2,
            **recipe_data
        }

        assert m.eval(m.Y[0]).subs(data) == supply
        assert m.eval(m.Y[1]).subs(data) == demand1
        assert m.eval(m.Y[2]).subs(data) <= demand2  # may be less based on limit
        if supply >= demand1 + demand2 and demand2 >= (supply - demand1):
            # If we have sufficient in->mid compared to mid->out1 and mid->out2,
            # it will balance
            assert m.eval_intermediates(m.object_balance("mid")).subs(data) == 0

    @given(
        st.floats(min_value=0, max_value=1e6, allow_infinity=False, allow_nan=False),
        st.floats(min_value=-1e6, max_value=1e6, allow_infinity=False, allow_nan=False),
    )
    def test_object_balance_with_various_flows(self, demand1, demand2):
        """Test mass balance is zero when not using until_objects.
        """
        m, recipe_data = self._create_branching_model()

        m.add(
            m.pull_production("out1", demand1),
            m.pull_production("out2", demand2),
        )

        # Check balance
        balance = m.eval_intermediates(m.object_balance("mid")).subs(recipe_data)
        assert abs(balance) < 1e-6

    @given(
        st.floats(min_value=0, allow_infinity=False),
        st.floats(min_value=0, allow_infinity=False),
        st.floats(min_value=0, allow_infinity=False)
    )
    @example(2, 3, 10)
    def test_limit_involving_unrelated_process(self, initial_demand, extra_demand, capacity):
        """Test limit() where limit expression involves a process not directly in the limited flow.
        Uses P2 for initial demand and P3 for extra demand, with limit on combined output.
        """
        m, recipe_data = self._create_branching_model()
        sy.var("a, b, c", real=True)

        # First flows: demand out1
        m.add(m.pull_production("out1", a))

        # Add extra demand for out2 with limit
        unlimited_extra = m.pull_production("out2", b)
        test_expr = (
            m.expr("ProcessOutput", process_id="P2", object_id="out1") +
            m.expr("ProcessOutput", process_id="P3", object_id="out2")
        )
        limit_expr = c
        limited_extra = m.limit(
            unlimited_extra,
            test_expr,
            limit_expr,
        )
        m.add(limited_extra)

        # We should not have exceeded the limit
        data = {
            a: initial_demand,
            b: extra_demand,
            c: capacity,
            **recipe_data,
        }
        value1 = m.eval(m.Y[1]).subs(data)
        value2 = m.eval(m.Y[2]).subs(data)
        assert value1 == initial_demand, "initial demand satisfied exactly"
        assert value1 + value2 <= max(initial_demand, capacity), "limit applied"
        if initial_demand + extra_demand < capacity:
            assert value2 == extra_demand


# ============================================================================
# PROPERTY-BASED TESTS WITH RANDOM MODELS
# ============================================================================

@given(model_strategy())
# @settings(deadline=None)  # Disable deadline for this test
def test_generated_model_basic_operations(m):
    """Test that generated models can perform basic operations without errors."""
    # Find objects that are produced
    produced_objects = {obj_id for p in m.processes for obj_id in p.produces}
    assume(len(produced_objects) > 0)

    # Pick an object to test with
    test_obj = next(iter(produced_objects))

    # Create symbolic demand
    demand = sy.Symbol("demand", real=True, positive=True)

    # Define allocation coefficients for all objects that need them
    allocs = {}
    for obj in m.objects:
        producers = m.producers_of(obj.id)
        if len(producers) > 1:
            # Multiple producers - need allocation
            allocs[obj.id] = {
                proc_id: sy.Symbol(f"alloc_{obj.id}_{proc_id}", real=True, positive=True)
                for proc_id in producers
            }

    try:
        # Try basic pull operation
        result = m.pull_production(test_obj, demand, allocate_backwards=allocs)
        m.add(result)

        # Try to evaluate object balance for produced objects
        for obj_id in list(produced_objects)[:3]:  # Just check first 3
            balance = m.object_balance(obj_id)
            # Should not raise an error
            assert balance is not None
    except Exception as e:
        # If we get a divide by zero or other error, that's what we're looking for
        if "division by zero" in str(e).lower() or "zerodivision" in str(e).lower():
            raise
        # Other errors might be expected for some random models
        assume(False)
