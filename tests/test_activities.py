"""Tests for the AdditionalActivity-based model building."""

import pytest
import sympy as sy
from rdflib import URIRef

from flowprog.model import ModelBuilder, Process, Object
from flowprog.activities import AdditionalActivity, Limit


MASS = URIRef("http://qudt.org/vocab/quantitykind/Mass")


def MObject(id, *args, **kwargs):
    return Object(id, MASS, *args, **kwargs)


a = sy.Symbol("a")
b = sy.Symbol("b")
c = sy.Symbol("c")


class TestAdditionalActivityFromPropagation:
    """Test that pull/push methods return AdditionalActivity with correct structure."""

    @pytest.fixture
    def m(self):
        processes = [Process("M1", produces=["out"], consumes=["in"])]
        objects = [MObject("in"), MObject("out")]
        return ModelBuilder(processes, objects)

    def test_pull_production_returns_additional_activity(self, m):
        result = m.pull_production("out", a)
        assert isinstance(result, AdditionalActivity)
        assert m.Y[0] in result.values
        assert m.X[0] in result.values
        assert result.transformations == []

    def test_push_consumption_returns_additional_activity(self, m):
        result = m.push_consumption("in", a)
        assert isinstance(result, AdditionalActivity)
        assert m.X[0] in result.values
        assert result.transformations == []

    def test_pull_production_intermediates_carried(self, m):
        result = m.pull_production("out", a)
        # Should have at least one intermediate (the value passed to pull_process_output)
        assert len(result.intermediates) > 0
        # Each intermediate is (symbol, expr, description)
        for sym, expr, desc in result.intermediates:
            assert isinstance(sym, sy.Symbol)
            assert isinstance(desc, str)


class TestAdditionalActivityWithAllocation:
    """Test AdditionalActivity with multiple producers requiring allocation."""

    @pytest.fixture
    def m(self):
        processes = [
            Process("M1", produces=["out"], consumes=["in1"]),
            Process("M2", produces=["out"], consumes=["in2"]),
        ]
        objects = [MObject("in1"), MObject("in2"), MObject("out")]
        return ModelBuilder(processes, objects)

    def test_allocated_pull_returns_additional_activity(self, m):
        a1, a2 = sy.symbols("a1 a2")
        result = m.pull_production(
            "out", a,
            allocate_backwards={"out": {"M1": a1, "M2": a2}},
        )
        assert isinstance(result, AdditionalActivity)
        # Both processes should have activity contributions
        assert m.X[0] in result.values
        assert m.X[1] in result.values


class TestLimit:
    """Test that limit() appends a Limit transformation."""

    @pytest.fixture
    def m(self):
        processes = [
            Process("P1", consumes=["in"], produces=["mid"]),
            Process("P2", consumes=["mid"], produces=["out"]),
        ]
        objects = [MObject("in"), MObject("mid", True), MObject("out")]
        return ModelBuilder(processes, objects)

    def test_limit_appends_transformation(self, m):
        proposal = m.pull_production("out", a)
        limited = m.limit(proposal, m.Y[0], c)
        assert isinstance(limited, AdditionalActivity)
        assert len(limited.transformations) == 1
        assert isinstance(limited.transformations[0], Limit)

    def test_limit_preserves_values(self, m):
        proposal = m.pull_production("out", a)
        limited = m.limit(proposal, m.Y[0], c)
        assert limited.values == proposal.values

    def test_limit_preserves_intermediates(self, m):
        proposal = m.pull_production("out", a)
        limited = m.limit(proposal, m.Y[0], c)
        assert limited.intermediates == proposal.intermediates

    def test_limit_does_not_mutate_original(self, m):
        proposal = m.pull_production("out", a)
        original_transformations = list(proposal.transformations)
        m.limit(proposal, m.Y[0], c)
        assert proposal.transformations == original_transformations

    def test_chained_limits(self, m):
        proposal = m.pull_production("out", a)
        limited1 = m.limit(proposal, m.Y[0], b)
        limited2 = m.limit(limited1, m.Y[0], c)
        assert len(limited2.transformations) == 2

    def test_limit_stores_raw_structural_symbols(self, m):
        """Limit should store raw X[j]/Y[j], not placeholders."""
        proposal = m.pull_production("out", a)
        limited = m.limit(proposal, m.Y[0], c)
        limit_transform = limited.transformations[0]
        # Expression should contain Y[0] directly
        assert m.Y[0] in limit_transform.expression.atoms(sy.Indexed)


class TestStructuralQuerySymbols:
    """Test that deficit/balance queries return structural indexed symbols."""

    @pytest.fixture
    def m(self):
        processes = [
            Process("M1", consumes=["in1"], produces=["mid"]),
            Process("M2", consumes=["in2"], produces=["mid"]),
            Process("Use", consumes=["mid"], produces=["out"]),
        ]
        objects = [MObject("in1"), MObject("in2"), MObject("mid"), MObject("out")]
        return ModelBuilder(processes, objects)

    def test_production_deficit_returns_indexed(self, m):
        result = m.object_production_deficit("mid")
        assert isinstance(result, sy.Indexed)
        assert result.base == m.structure.ProductionDeficit

    def test_same_deficit_returns_same_symbol(self, m):
        s1 = m.object_production_deficit("mid")
        s2 = m.object_production_deficit("mid")
        assert s1 == s2

    def test_different_objects_return_different_symbols(self, m):
        s1 = m.object_production_deficit("mid")
        s2 = m.object_production_deficit("out")
        assert s1 != s2

    def test_deficit_usable_in_arithmetic(self, m):
        deficit = m.object_production_deficit("mid")
        expr = deficit / 2 + 1
        assert deficit in expr.atoms(sy.Indexed)

    def test_consumption_deficit_returns_indexed(self, m):
        result = m.object_consumption_deficit("mid")
        assert isinstance(result, sy.Indexed)
        assert result.base == m.structure.ConsumptionDeficit

    def test_production_and_consumption_deficit_different(self, m):
        p = m.object_production_deficit("mid")
        c = m.object_consumption_deficit("mid")
        assert p != c

    def test_balance_returns_indexed(self, m):
        result = m.object_balance("mid")
        assert isinstance(result, sy.Indexed)
        assert result.base == m.structure.Balance


class TestAddAndCompile:
    """Test that add() records steps and build() compiles to correct results."""

    @pytest.fixture
    def m(self):
        processes = [Process("M1", produces=["out"], consumes=["in"])]
        objects = [MObject("in"), MObject("out")]
        return ModelBuilder(processes, objects)

    def test_add_records_step(self, m):
        activity = m.pull_production("out", a)
        m.add(activity, label="test step")
        assert len(m._steps) == 1
        assert m._steps[0].description == "test step"

    def test_add_accepts_bare_dict(self, m):
        """Backward compatibility: add() should accept bare dicts."""
        m.add({m.X[0]: a}, label="bare dict")
        assert len(m._steps) == 1

    def test_compiled_model_evaluates_correctly(self, m):
        """End-to-end: build and evaluate a simple model."""
        m.add(m.pull_production("out", a), label="demand")
        model = m.build()
        result = model.eval(model.Y[0])
        assert result == a / m.S[1, 0]

    def test_two_adds_accumulate(self, m):
        m.add(m.pull_production("out", a), label="first")
        m.add(m.pull_production("out", b), label="second")
        model = m.build()
        result = model.eval(model.Y[0])
        assert sy.simplify(result - (a + b) / m.S[1, 0]) == 0


class TestDeficitCompilation:
    """Test that deficit symbols compile correctly."""

    @pytest.fixture
    def m(self):
        processes = [
            Process("M1", consumes=["in1"], produces=["mid"]),
            Process("M2", consumes=["in2"], produces=["mid"]),
            Process("Use", consumes=["mid"], produces=["out"]),
        ]
        objects = [MObject("in1"), MObject("in2"), MObject("mid"), MObject("out")]
        return ModelBuilder(processes, objects)

    def test_deficit_resolved_during_compilation(self, m):
        """Production deficit should reflect previously committed state."""
        # Pull demand from one side
        m.add(m.pull_production("out", a, until_objects=["mid"]), label="demand")
        m.add(m.push_consumption("in1", b, until_objects=["mid"]), label="supply")

        # Balance using deficit
        m.add(
            m.pull_process_output("M2", "mid", m.object_production_deficit("mid")),
            label="balance",
        )

        model = m.build()

        # M2's activity should equal the deficit: demand - supply
        # Same as the existing test_balance_mid_object test
        assert model.eval(model.X[0]) == b / m.U[0, 0]
        assert model.eval(model.X[2]) == a / m.S[3, 2]
        assert (
            model.eval(model.X[1])
            == sy.Max(0, a / m.S[3, 2] * m.U[2, 2] - b / m.U[0, 0] * m.S[2, 0])
            / m.S[2, 1]
        )


class TestLimitCompilation:
    """Test that Limit transformations compile to correct constrained values."""

    @pytest.fixture
    def simple(self):
        processes = [Process("P1", consumes=["in"], produces=["out"])]
        objects = [MObject("in"), MObject("out")]
        return ModelBuilder(processes, objects)

    @pytest.fixture
    def chain(self):
        processes = [
            Process("P1", consumes=["in"], produces=["mid"]),
            Process("P2", consumes=["mid"], produces=["out"]),
        ]
        objects = [MObject("in"), MObject("mid", True), MObject("out")]
        return ModelBuilder(processes, objects)

    def test_limit_respects_capacity(self, chain):
        """Compiled limit should not exceed the capacity.

        Uses the same test setup as the existing test_limit_with_symbols property test.
        """
        m = chain
        # First demand
        m.add(m.pull_production("out", a), label="initial")

        # Extra demand, limited
        extra = m.pull_production("out", b)
        m.add(m.limit(extra, m.Y[0], c), label="limited extra")

        model = m.build()

        recipe = {
            m.U[0, 0]: 1.0, m.S[1, 0]: 0.6,
            m.U[1, 1]: 2.2, m.S[2, 1]: 2.0,
        }

        # initial=10 gives Y[0] = 10/2.0*2.2/0.6 = 18.33
        # With limit=100 (well above), extra should pass through fully
        value_unlimited = model.eval(model.Y[0]).subs({a: 10, b: 5, c: 100, **recipe})
        initial_y0 = 10 / 2.0 * 2.2 / 0.6
        extra_y0 = 5 / 2.0 * 2.2 / 0.6
        assert abs(float(value_unlimited) - (initial_y0 + extra_y0)) < 1e-9

        # With limit=20 (between initial and initial+extra), extra should be clamped
        value_limited = model.eval(model.Y[0]).subs({a: 10, b: 5, c: 20, **recipe})
        assert float(value_limited) <= 20.0 + 1e-9
        assert float(value_limited) >= initial_y0 - 1e-9
