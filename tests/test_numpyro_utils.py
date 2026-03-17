"""Tests for the NumPyro compiler utils.
"""

import jax.numpy as jnp
import sympy as sy

from flowprog.model_builder import ModelBuilder, Process, Object, ModelStructure
from flowprog.backends.numpyro.numpyro_compiler import (
    _normalize_recipe,
    _collect_free_parameters,
)
from flowprog.backends.numpyro.sympy_to_jax import sympy_expr_to_jax
from flowprog.backends.numpyro.utils import smooth_clamp, softplus, smooth_max


class TestSmoothUtils:
    """Tests for smooth approximation utilities."""

    def test_softplus_approx_max_zero(self):
        """softplus(x) ≈ max(0, x) for large beta."""
        for x in [-5.0, -1.0, 0.0, 1.0, 5.0]:
            result = float(softplus(jnp.float64(x), beta=1000.0))
            expected = max(0.0, x)
            assert (
                abs(result - expected) < 0.01
            ), f"softplus({x}) = {result}, expected ≈ {expected}"

    def test_softplus_smooth_near_zero(self):
        """softplus should be smooth (nonzero) near x=0."""
        result = float(softplus(jnp.float64(0.0), beta=100.0))
        assert result > 0.0  # Should be slightly positive, not exactly zero

    def test_smooth_max_approx(self):
        """smooth_max(a, b) ≈ max(a, b) for large beta."""
        pairs = [(1.0, 3.0), (-2.0, 0.5), (0.0, 0.0), (5.0, 5.0)]
        for a, b in pairs:
            result = float(smooth_max(jnp.float64(a), jnp.float64(b), beta=1000.0))
            expected = max(a, b)
            assert (
                abs(result - expected) < 0.01
            ), f"smooth_max({a}, {b}) = {result}, expected ≈ {expected}"

    def test_smooth_clamp(self):
        """smooth_clamp(x, lo, hi) ≈ clamp(x, lo, hi) for large beta."""
        cases = [
            (-1.0, 0.0, 1.0, 0.0),
            (0.5, 0.0, 1.0, 0.5),
            (2.0, 0.0, 1.0, 1.0),
        ]
        for x, lo, hi, expected in cases:
            result = float(
                smooth_clamp(
                    jnp.float64(x), jnp.float64(lo), jnp.float64(hi), beta=1000.0
                )
            )
            assert (
                abs(result - expected) < 0.01
            ), f"smooth_clamp({x}, {lo}, {hi}) = {result}, expected ≈ {expected}"


class TestRecipeNormalization:
    """Tests for recipe data format conversion."""

    def test_id_based_recipe(self):
        """ID-based recipe data is correctly converted to sympy-indexed format."""
        builder = ModelBuilder(
            processes=[Process("P1", consumes=["in"], produces=["out"])],
            objects=[Object("out", metric="Mass"), Object("in", metric="Mass")],
        )
        recipe_data = {
            "P1": {"produces": {"out": 1.0}, "consumes": {"in": 0.8}},
        }
        result = _normalize_recipe(builder.structure, recipe_data)

        S = builder.S
        U = builder.U

        assert result == {
            S[0, 0]: 1.0,
            U[1, 0]: 0.8,
        }


class TestFreeParameterCollection:
    """Tests for free parameter detection."""

    def test_collect_free_params(self):
        builder = ModelBuilder(
            processes=[Process("P1", consumes=[], produces=["out"])],
            objects=[Object("out", metric="Mass")],
        )
        k = sy.Symbol("k", positive=True)

        # Create a step with a ProductionDeficit symbol:
        # k * ProductionDeficit — this expression's free_symbols
        # includes Symbol('ProductionDeficit') as the IndexedBase label.
        deficit = builder.object_production_deficit("out")
        step = builder.pull_process_output("P1", "out", k * deficit)

        free_params = _collect_free_parameters(builder.structure, [step])
        assert free_params == {"k": k}

        # Note that structural base labels must NOT appear IndexedBase labels
        # (Balance, ProductionDeficit, etc.) must not appear as free parameters
        # when a step multiplies a free symbol by a structural symbol like
        # ProductionDeficit[i].


class TestExpressionWalker:
    """Tests for the SymPy-to-JAX expression walker."""

    def test_arithmetic(self):
        """Basic arithmetic expressions are correctly converted."""
        structure = ModelStructure([], [])

        a, b = sy.symbols("a b")
        env = {a: jnp.float64(3.0), b: jnp.float64(4.0)}

        # a + b
        result = float(sympy_expr_to_jax(a + b, env, structure, {}))
        assert abs(result - 7.0) < 1e-10

        # a * b
        result = float(sympy_expr_to_jax(a * b, env, structure, {}))
        assert abs(result - 12.0) < 1e-10

        # a / b
        result = float(sympy_expr_to_jax(a / b, env, structure, {}))
        assert abs(result - 0.75) < 1e-10

        # a ** 2
        result = float(sympy_expr_to_jax(a**2, env, structure, {}))
        assert abs(result - 9.0) < 1e-10

    def test_constant_lookup(self):
        """Constants (recipe values) are correctly resolved from the constants dict."""
        structure = ModelStructure(
            processes=[Process("P1", consumes=["in"], produces=["out"])],
            objects=[Object("out", metric="Mass"), Object("in", metric="Mass")],
        )
        S = structure.S
        U = structure.U
        constants = {S[0, 0]: 1.0, U[1, 0]: 0.8}

        result = float(sympy_expr_to_jax(S[0, 0], {}, structure, constants))
        assert abs(result - 1.0) < 1e-10

        result = float(sympy_expr_to_jax(U[1, 0], {}, structure, constants))
        assert abs(result - 0.8) < 1e-10

    def test_env_takes_priority_over_constants(self):
        """When the same symbol is in both env and constants, env wins."""
        structure = ModelStructure(
            processes=[Process("P1", consumes=[], produces=["out"])],
            objects=[Object("out", metric="Mass")],
        )

        S = structure.S
        constants = {S[0, 0]: 1.0}
        env = {S[0, 0]: jnp.float64(42.0)}

        result = float(sympy_expr_to_jax(S[0, 0], env, structure, constants))
        assert abs(result - 42.0) < 1e-10

    def test_max_to_softplus(self):
        """Max(0, x) is converted to softplus(x)."""
        structure = ModelStructure([], [])

        x = sy.Symbol("x")
        env = {x: jnp.float64(5.0)}

        expr = sy.Max(0, x, evaluate=False)
        result = float(sympy_expr_to_jax(expr, env, structure, {}))
        assert abs(result - 5.0) < 0.1  # softplus(5) ≈ 5

        env = {x: jnp.float64(-5.0)}
        result = float(sympy_expr_to_jax(expr, env, structure, {}))
        assert abs(result - 0.0) < 0.1  # softplus(-5) ≈ 0
