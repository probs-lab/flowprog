"""
Tests for MILP transformation components.

Tests cover:
- BoundsAnalyzer: Computing bounds on expressions
- PiecewiseLinearizer: Linearizing piecewise operations
- MILPTransformer: Full model transformation
- Integration: Round-trip through transformation and solving
"""

import pytest
import sympy as sy
from flowprog.milp_transform import (
    BoundsAnalyzer,
    PiecewiseLinearizer,
    MILPTransformer,
    MILPModel,
    MILPVariable,
    MILPConstraint,
    ConstraintType,
    MILPObjective,
    QuadraticTerm,
)


class TestBoundsAnalyzer:
    """Test bounds computation on symbolic expressions."""

    def test_constant_bounds(self):
        """Test bounds on constant expressions."""
        analyzer = BoundsAnalyzer({})

        lower, upper = analyzer.get_bounds(sy.Integer(5))
        assert lower == 5.0
        assert upper == 5.0

    def test_symbol_bounds(self):
        """Test bounds on symbols with known ranges."""
        x = sy.Symbol("x")
        y = sy.Symbol("y")

        analyzer = BoundsAnalyzer({
            x: (0.0, 10.0),
            y: (-5.0, 5.0),
        })

        lower, upper = analyzer.get_bounds(x)
        assert lower == 0.0
        assert upper == 10.0

        lower, upper = analyzer.get_bounds(y)
        assert lower == -5.0
        assert upper == 5.0

    def test_addition_bounds(self):
        """Test bounds on addition."""
        x = sy.Symbol("x")
        y = sy.Symbol("y")

        analyzer = BoundsAnalyzer({
            x: (0.0, 10.0),
            y: (5.0, 15.0),
        })

        expr = x + y
        lower, upper = analyzer.get_bounds(expr)

        assert lower == 5.0  # 0 + 5
        assert upper == 25.0  # 10 + 15

    def test_multiplication_bounds(self):
        """Test bounds on multiplication."""
        x = sy.Symbol("x")
        y = sy.Symbol("y")

        analyzer = BoundsAnalyzer({
            x: (2.0, 5.0),
            y: (3.0, 4.0),
        })

        expr = x * y
        lower, upper = analyzer.get_bounds(expr)

        assert lower == 6.0   # 2 * 3
        assert upper == 20.0  # 5 * 4

    def test_multiplication_with_negatives(self):
        """Test bounds on multiplication with negative values."""
        x = sy.Symbol("x")
        y = sy.Symbol("y")

        analyzer = BoundsAnalyzer({
            x: (-5.0, 5.0),
            y: (-2.0, 3.0),
        })

        expr = x * y
        lower, upper = analyzer.get_bounds(expr)

        # Possible products: (-5)*(-2)=10, (-5)*3=-15, 5*(-2)=-10, 5*3=15
        assert lower == -15.0
        assert upper == 15.0

    def test_max_bounds(self):
        """Test bounds on max expression."""
        x = sy.Symbol("x")
        y = sy.Symbol("y")

        analyzer = BoundsAnalyzer({
            x: (0.0, 10.0),
            y: (5.0, 15.0),
        })

        expr = sy.Max(x, y, evaluate=False)
        lower, upper = analyzer.get_bounds(expr)

        assert lower == 5.0   # max(0, 5) = 5
        assert upper == 15.0  # max(10, 15) = 15

    def test_min_bounds(self):
        """Test bounds on min expression."""
        x = sy.Symbol("x")
        y = sy.Symbol("y")

        analyzer = BoundsAnalyzer({
            x: (0.0, 10.0),
            y: (5.0, 15.0),
        })

        expr = sy.Min(x, y, evaluate=False)
        lower, upper = analyzer.get_bounds(expr)

        assert lower == 0.0   # min(0, 5) = 0
        assert upper == 10.0  # min(10, 15) = 10

    def test_abs_bounds(self):
        """Test bounds on absolute value."""
        x = sy.Symbol("x")

        # Case 1: All positive
        analyzer = BoundsAnalyzer({x: (2.0, 5.0)})
        expr = sy.Abs(x)
        lower, upper = analyzer.get_bounds(expr)
        assert lower == 2.0
        assert upper == 5.0

        # Case 2: All negative
        analyzer = BoundsAnalyzer({x: (-5.0, -2.0)})
        expr = sy.Abs(x)
        lower, upper = analyzer.get_bounds(expr)
        assert lower == 2.0
        assert upper == 5.0

        # Case 3: Crossing zero
        analyzer = BoundsAnalyzer({x: (-3.0, 5.0)})
        expr = sy.Abs(x)
        lower, upper = analyzer.get_bounds(expr)
        assert lower == 0.0
        assert upper == 5.0  # max(abs(-3), abs(5))

    def test_piecewise_bounds(self):
        """Test bounds on piecewise expression."""
        x = sy.Symbol("x")

        analyzer = BoundsAnalyzer({x: (0.0, 10.0)})

        # Piecewise with two branches
        expr = sy.Piecewise(
            (x, x > 5),
            (2 * x, True),
            evaluate=False
        )

        lower, upper = analyzer.get_bounds(expr)

        # Conservative: min(0, 0) to max(10, 20)
        assert lower == 0.0
        assert upper == 20.0

    def test_big_m_computation(self):
        """Test big-M value computation."""
        x = sy.Symbol("x")

        analyzer = BoundsAnalyzer({x: (-10.0, 20.0)})

        expr = 2 * x + 5
        M = analyzer.compute_big_m(expr)

        # Bounds: 2*(-10)+5 = -15, 2*20+5 = 45
        # max(abs(-15), abs(45)) = 45
        # With slack 1.1: 49.5
        assert M == pytest.approx(49.5, rel=0.01)

    def test_unbounded_expression(self):
        """Test handling of unbounded expressions."""
        x = sy.Symbol("x")

        # No bounds provided
        analyzer = BoundsAnalyzer({})

        lower, upper = analyzer.get_bounds(x)
        assert lower == float('-inf')
        assert upper == float('inf')

        # big-M should fallback to large finite value
        M = analyzer.compute_big_m(x)
        assert M == 1e6


class TestPiecewiseLinearizer:
    """Test linearization of piecewise operations."""

    def test_max_zero_linearization(self):
        """Test linearization of max(0, linear_expr)."""
        x = sy.Symbol("x")

        analyzer = BoundsAnalyzer({x: (0.0, 10.0)})
        linearizer = PiecewiseLinearizer(analyzer)

        milp_model = MILPModel()
        milp_model.variable_mapping[x] = "x"
        milp_model.add_variable(MILPVariable(name="x", lower_bound=0.0, upper_bound=10.0))

        # max(0, x - 5)
        expr = x - 5
        coeffs = {x: 1.0}
        constant = -5.0

        result_var = linearizer.linearize_max_zero(expr, coeffs, constant, milp_model)

        # Should create auxiliary variable and binary indicator
        assert result_var in milp_model.variables
        assert milp_model.variables[result_var].is_auxiliary

        # Should create 4 constraints
        constraint_names = {c.name for c in milp_model.constraints}
        assert any("ge_0" in name for name in constraint_names)
        assert any("ge_expr" in name for name in constraint_names)

        # Should have exactly one binary variable for this operation
        binary_vars = [v for v in milp_model.variables.values() if v.is_binary]
        assert len(binary_vars) >= 1

    def test_max_two_linearization(self):
        """Test linearization of max(expr1, expr2)."""
        x = sy.Symbol("x")
        y = sy.Symbol("y")

        analyzer = BoundsAnalyzer({
            x: (0.0, 10.0),
            y: (0.0, 15.0),
        })
        linearizer = PiecewiseLinearizer(analyzer)

        milp_model = MILPModel()
        milp_model.variable_mapping[x] = "x"
        milp_model.variable_mapping[y] = "y"
        milp_model.add_variable(MILPVariable(name="x", lower_bound=0.0, upper_bound=10.0))
        milp_model.add_variable(MILPVariable(name="y", lower_bound=0.0, upper_bound=15.0))

        # max(x + 2, y - 3)
        expr1 = x + 2
        expr2 = y - 3
        coeffs1 = {x: 1.0}
        constant1 = 2.0
        coeffs2 = {y: 1.0}
        constant2 = -3.0

        result_var = linearizer.linearize_max_two(
            expr1, expr2,
            coeffs1, constant1,
            coeffs2, constant2,
            milp_model
        )

        assert result_var in milp_model.variables
        assert milp_model.variables[result_var].is_auxiliary

        # Should create 4 constraints (2 lower bounds, 2 conditional upper bounds)
        assert len(milp_model.constraints) >= 4

    def test_min_two_linearization(self):
        """Test linearization of min(expr1, expr2)."""
        x = sy.Symbol("x")
        y = sy.Symbol("y")

        analyzer = BoundsAnalyzer({
            x: (0.0, 10.0),
            y: (0.0, 15.0),
        })
        linearizer = PiecewiseLinearizer(analyzer)

        milp_model = MILPModel()
        milp_model.variable_mapping[x] = "x"
        milp_model.variable_mapping[y] = "y"
        milp_model.add_variable(MILPVariable(name="x", lower_bound=0.0, upper_bound=10.0))
        milp_model.add_variable(MILPVariable(name="y", lower_bound=0.0, upper_bound=15.0))

        expr1 = x
        expr2 = y
        coeffs1 = {x: 1.0}
        constant1 = 0.0
        coeffs2 = {y: 1.0}
        constant2 = 0.0

        result_var = linearizer.linearize_min_two(
            expr1, expr2,
            coeffs1, constant1,
            coeffs2, constant2,
            milp_model
        )

        assert result_var in milp_model.variables

        # Min creates negated max, plus equality constraint
        # So should have more constraints than just max
        assert len(milp_model.constraints) >= 5

    def test_abs_linearization(self):
        """Test linearization of abs(linear_expr)."""
        x = sy.Symbol("x")

        analyzer = BoundsAnalyzer({x: (-10.0, 10.0)})
        linearizer = PiecewiseLinearizer(analyzer)

        milp_model = MILPModel()
        milp_model.variable_mapping[x] = "x"
        milp_model.add_variable(MILPVariable(name="x", lower_bound=-10.0, upper_bound=10.0))

        # abs(x - 5)
        expr = x - 5
        coeffs = {x: 1.0}
        constant = -5.0

        result_var = linearizer.linearize_abs(expr, coeffs, constant, milp_model)

        assert result_var in milp_model.variables

        # Abs is implemented as max(expr, -expr), which creates max constraints
        assert len(milp_model.constraints) >= 4


class TestMILPModel:
    """Test MILP model data structure."""

    def test_add_variable(self):
        """Test adding variables to model."""
        model = MILPModel()

        var = MILPVariable(
            name="x",
            lower_bound=0.0,
            upper_bound=10.0,
            description="Test variable"
        )

        name = model.add_variable(var)
        assert name == "x"
        assert "x" in model.variables

    def test_add_constraint(self):
        """Test adding constraints to model."""
        model = MILPModel()

        constraint = MILPConstraint(
            name="c1",
            coefficients={"x": 1.0, "y": 2.0},
            constraint_type=ConstraintType.LINEAR_LE,
            rhs=10.0,
            description="x + 2y <= 10"
        )

        model.add_constraint(constraint)
        assert len(model.constraints) == 1
        assert model.constraints[0].name == "c1"

    def test_objective_linear_terms(self):
        """Test adding linear terms to objective."""
        obj = MILPObjective()

        obj.add_linear_term("x", 2.0)
        obj.add_linear_term("y", 3.0)
        obj.add_linear_term("x", 1.0)  # Should accumulate

        assert obj.linear_terms["x"] == 3.0
        assert obj.linear_terms["y"] == 3.0

    def test_objective_quadratic_terms(self):
        """Test adding quadratic terms to objective."""
        obj = MILPObjective()

        obj.add_quadratic_term("x", "x", 1.0)  # x^2
        obj.add_quadratic_term("x", "y", 2.0)  # 2xy

        assert len(obj.quadratic_terms) == 2
        assert obj.quadratic_terms[0].var1 == "x"
        assert obj.quadratic_terms[0].var2 == "x"
        assert obj.quadratic_terms[0].coefficient == 1.0

    def test_model_statistics(self):
        """Test model statistics computation."""
        model = MILPModel()

        # Add some variables
        model.add_variable(MILPVariable(name="x", lower_bound=0.0))
        model.add_variable(MILPVariable(name="y", is_binary=True))
        model.add_variable(MILPVariable(name="z", lower_bound=0.0))

        # Add constraint
        model.add_constraint(MILPConstraint(
            name="c1",
            coefficients={"x": 1.0},
            constraint_type=ConstraintType.LINEAR_EQ,
            rhs=5.0
        ))

        # Add objective
        model.objective.add_linear_term("x", 1.0)
        model.objective.add_quadratic_term("x", "x", 1.0)

        stats = model.get_statistics()

        assert stats["total_variables"] == 3
        assert stats["continuous_variables"] == 2
        assert stats["binary_variables"] == 1
        assert stats["constraints"] == 1
        assert stats["linear_objective_terms"] == 1
        assert stats["quadratic_objective_terms"] == 1


class TestMILPTransformer:
    """Test full MILP transformation of flowprog models."""

    def test_simple_max_transformation(self):
        """Test transformation of simple model with max operation."""
        from flowprog.imperative_model import Model, Process, Object
        from rdflib import URIRef

        # Create minimal model
        processes = [Process(id="P1", produces=["O1"], consumes=[])]
        objects = [Object(id="O1", metric=URIRef("http://example.org/metric"))]

        model = Model(processes=processes, objects=objects)

        # Create simple expression with max
        x = sy.Symbol("x", nonnegative=True)
        demand = sy.Max(0, x - 5, evaluate=False)

        # Simple pull operation
        model.add(model.pull_production("O1", demand, until_objects=[]))

        # Transform
        transformer = MILPTransformer(model)

        objective_targets = {
            model.expr("SoldProduction", object_id="O1"): 10.0
        }

        variable_bounds = {
            x: (0.0, 20.0),
            model.X[0]: (0.0, 20.0),
            model.Y[0]: (0.0, 20.0),
        }

        # Provide recipe coefficients (S and U matrices)
        fixed_values = {
            model.S[0, 0]: 1.0,  # Process 0 produces 1 unit of object 0 per Y[0]
        }

        milp_model = transformer.transform(
            objective_targets=objective_targets,
            variable_bounds=variable_bounds,
            fixed_values=fixed_values
        )

        # Check that transformation succeeded
        assert milp_model is not None
        assert len(milp_model.variables) > 0

        # Should have created binary variables for max operation
        binary_vars = [v for v in milp_model.variables.values() if v.is_binary]
        assert len(binary_vars) >= 1

        # Should have quadratic objective (squared error)
        assert len(milp_model.objective.quadratic_terms) >= 1

    def test_transformation_preserves_variable_mapping(self):
        """Test that transformation maintains mapping from sympy to MILP variables."""
        from flowprog.imperative_model import Model, Process, Object
        from rdflib import URIRef

        processes = [Process(id="P1", produces=["O1"], consumes=[])]
        objects = [Object(id="O1", metric=URIRef("http://example.org/metric"))]

        model = Model(processes=processes, objects=objects)

        x = sy.Symbol("x", nonnegative=True)
        model.add(model.pull_production("O1", x, until_objects=[]))

        transformer = MILPTransformer(model)

        objective_targets = {
            model.expr("SoldProduction", object_id="O1"): 10.0
        }

        variable_bounds = {
            x: (0.0, 20.0),
            model.X[0]: (0.0, 20.0),
            model.Y[0]: (0.0, 20.0),
        }

        # Provide recipe coefficients
        fixed_values = {
            model.S[0, 0]: 1.0,
        }

        milp_model = transformer.transform(
            objective_targets=objective_targets,
            variable_bounds=variable_bounds,
            fixed_values=fixed_values
        )

        # Check that mapping exists
        assert x in milp_model.variable_mapping
        assert model.X[0] in milp_model.variable_mapping
        assert model.Y[0] in milp_model.variable_mapping

        # Check that mapped variables exist in MILP model
        x_milp_name = milp_model.variable_mapping[x]
        assert x_milp_name in milp_model.variables


def test_constraint_expression_formatting():
    """Test that constraints can be formatted as human-readable expressions."""
    constraint = MILPConstraint(
        name="test",
        coefficients={"x": 2.0, "y": -1.0, "z": 3.5},
        constraint_type=ConstraintType.LINEAR_LE,
        rhs=10.0
    )

    expr = constraint.to_expression()

    # Should contain variable names and coefficients
    assert "x" in expr
    assert "y" in expr
    assert "z" in expr
    assert "<=" in expr
    assert "10.0" in expr
