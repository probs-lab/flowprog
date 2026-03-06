"""Tests for the NumPyro compiler.

Tests the three-step recycling model from Section 2.3.2 of the paper:
- 5 objects (O1–O5), 3 processes (P1–P3)
- P1 consumes O2 and O3, produces O1
- P2 consumes O4, produces O2
- P3 consumes O5, produces O2

Step 1: Pull production of O1 = f, stopping at O2
Step 2: Push consumption of O4 = g, stopping at O2
Step 3: Pull production deficit of O2 from P3

Expected (sharp):
  P1 activity: f / S₁₁
  P2 activity: g / U₄₂
  P3 activity: max(0, f·U₂₁/S₁₁ − g·S₂₂/U₄₂) / S₂₃
"""

import os
os.environ.setdefault("JAX_ENABLE_X64", "True")

import pytest
import jax
import jax.numpy as jnp
import numpyro
import numpyro.handlers as handlers
import numpyro.distributions as dist
import numpyro.infer
import sympy as sy

from flowprog.model import ModelBuilder, Process, Object
from flowprog.compilers.numpyro import (
    compile_numpyro, Observation, ModelSpec, _normalize_recipe,
)
from flowprog.compilers.numpyro_utils import smooth_clamp, softplus, smooth_max
from flowprog.compilers.sympy import compile_sympy

from .model_strategies import MObject


# ── Smooth utility tests ────────────────────────────────────────────


class TestSmoothUtils:
    """Tests for smooth approximation utilities."""

    def test_softplus_approx_max_zero(self):
        """softplus(x) ≈ max(0, x) for large beta."""
        for x in [-5.0, -1.0, 0.0, 1.0, 5.0]:
            result = float(softplus(jnp.float64(x), beta=1000.0))
            expected = max(0.0, x)
            assert abs(result - expected) < 0.01, f"softplus({x}) = {result}, expected ≈ {expected}"

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
            assert abs(result - expected) < 0.01, f"smooth_max({a}, {b}) = {result}, expected ≈ {expected}"

    def test_smooth_clamp(self):
        """smooth_clamp(x, lo, hi) ≈ clamp(x, lo, hi) for large beta."""
        cases = [
            (-1.0, 0.0, 1.0, 0.0),
            (0.5, 0.0, 1.0, 0.5),
            (2.0, 0.0, 1.0, 1.0),
        ]
        for x, lo, hi, expected in cases:
            result = float(smooth_clamp(
                jnp.float64(x), jnp.float64(lo), jnp.float64(hi), beta=1000.0
            ))
            assert abs(result - expected) < 0.01, (
                f"smooth_clamp({x}, {lo}, {hi}) = {result}, expected ≈ {expected}"
            )


# ── Three-step recycling model fixtures ──────────────────────────────


def _make_recycling_structure():
    """Create the five-object, three-process recycling model structure.

    Objects: O1, O2, O3, O4, O5
    Processes:
      P1: consumes O2, O3 → produces O1
      P2: consumes O4 → produces O2
      P3: consumes O5 → produces O2
    """
    processes = [
        Process("P1", produces=["O1"], consumes=["O2", "O3"]),
        Process("P2", produces=["O2"], consumes=["O4"]),
        Process("P3", produces=["O2"], consumes=["O5"]),
    ]
    objects = [
        MObject("O1"),
        MObject("O2", has_market=True),
        MObject("O3"),
        MObject("O4"),
        MObject("O5"),
    ]
    return processes, objects


RECIPE_DATA = {
    "P1": {"produces": {"O1": 1.0}, "consumes": {"O2": 0.8, "O3": 0.3}},
    "P2": {"produces": {"O2": 1.0}, "consumes": {"O4": 1.0}},
    "P3": {"produces": {"O2": 1.0}, "consumes": {"O5": 0.5}},
}


def _build_recycling_model(f_val, g_val):
    """Build the three-step recycling model using ModelBuilder.

    Step 1: Pull production of O1 = f, stopping at O2
    Step 2: Push consumption of O4 = g, stopping at O2
    Step 3: Pull production deficit of O2 from P3

    Returns (builder, f_sym, g_sym) where f_sym and g_sym are the sympy symbols.
    """
    processes, objects = _make_recycling_structure()
    builder = ModelBuilder(processes, objects)

    f = sy.Symbol("f", positive=True)
    g = sy.Symbol("g", positive=True)

    # Step 1: Pull production of O1 = f, stopping at O2
    step1 = builder.pull_production("O1", f, until_objects={"O2"})
    builder.add(step1, label="demand")

    # Step 2: Push consumption of O4 = g, stopping at O2
    step2 = builder.push_consumption("O4", g, until_objects={"O2"})
    builder.add(step2, label="recyclate supply")

    # Step 3: Pull production deficit of O2 from P3 specifically
    deficit = builder.object_production_deficit("O2")
    step3 = builder.pull_process_output("P3", "O2", deficit, until_objects={"O2"})
    builder.add(step3, label="deficit production")

    return builder, f, g


def _eval_sympy_model(builder, f_val, g_val):
    """Evaluate the sympy-compiled model for given f and g values.

    Returns dict of {symbol_str: float_value} for X[0], X[1], X[2], Y[0], Y[1], Y[2].
    """
    model = builder.build(RECIPE_DATA)
    f = sy.Symbol("f", positive=True)
    g = sy.Symbol("g", positive=True)
    params = {f: f_val, g: g_val}

    results = {}
    S = builder.structure
    for j in range(3):
        x_val = model.eval(S.X[j], params)
        y_val = model.eval(S.Y[j], params)
        results[f"X_{j}"] = float(x_val)
        results[f"Y_{j}"] = float(y_val)
    return results


def _eval_numpyro_model(builder, f_val, g_val, beta=1000.0):
    """Evaluate the numpyro-compiled model for given f and g.

    Passes f and g as keyword arguments to the model function.
    """
    model_fn = compile_numpyro(
        builder.structure,
        builder._steps,
        recipe_data=RECIPE_DATA,
        beta=beta,
    )

    # Run the model with fixed parameter values (passed as kwargs)
    with handlers.seed(rng_seed=0):
        trace = handlers.trace(model_fn).get_trace(f=f_val, g=g_val)

    results = {}
    for j in range(3):
        proc_id = builder.structure.processes[j].id
        x_key = f"X_{j}_{proc_id}"
        y_key = f"Y_{j}_{proc_id}"
        if x_key in trace:
            results[f"X_{j}"] = float(trace[x_key]["value"])
        if y_key in trace:
            results[f"Y_{j}"] = float(trace[y_key]["value"])
    return results


# ── Tests ────────────────────────────────────────────────────────────


class TestRecipeNormalization:
    """Tests for recipe data format conversion."""

    def test_id_based_recipe(self):
        """ID-based recipe data is correctly converted to sympy-indexed format."""
        processes, objects = _make_recycling_structure()
        builder = ModelBuilder(processes, objects)
        result = _normalize_recipe(builder.structure, RECIPE_DATA)
        S = builder.structure.S
        U = builder.structure.U

        assert result[S[0, 0]] == 1.0
        assert result[U[1, 0]] == 0.8
        assert result[U[2, 0]] == 0.3
        assert result[S[1, 1]] == 1.0
        assert result[U[3, 1]] == 1.0
        assert result[S[1, 2]] == 1.0
        assert result[U[4, 2]] == 0.5


class TestFreeParameterCollection:
    """Tests for free parameter detection."""

    def test_finds_free_params(self):
        """Free parameters (f, g) are detected from step expressions."""
        from flowprog.compilers.numpyro import _collect_free_parameters

        builder, f, g = _build_recycling_model(5, 1)
        params = _collect_free_parameters(builder.structure, builder._steps)
        assert "f" in params
        assert "g" in params

    def test_structural_symbol_labels_not_leaked(self):
        """IndexedBase labels (Balance, ProductionDeficit, etc.) must not
        appear as free parameters when a step multiplies a free symbol by
        a structural symbol like ProductionDeficit[i]."""
        from flowprog.compilers.numpyro import _collect_free_parameters

        processes, objects = _make_recycling_structure()
        builder = ModelBuilder(processes, objects)
        k = sy.Symbol("k", positive=True)
        f = sy.Symbol("f", positive=True)

        # Step 1: pull demand
        step1 = builder.pull_production("O1", f, until_objects={"O2"})
        builder.add(step1, label="demand")

        # Step 2: k * ProductionDeficit — this expression's free_symbols
        # includes Symbol('ProductionDeficit') as the IndexedBase label.
        deficit = builder.object_production_deficit("O2")
        step2 = builder.pull_process_output(
            "P2", "O2", k * deficit, until_objects={"O2"}
        )
        builder.add(step2, label="P2 share")

        params = _collect_free_parameters(builder.structure, builder._steps)
        assert "f" in params
        assert "k" in params
        # Structural base labels must NOT appear
        for name in ("ProductionDeficit", "Balance", "ConsumptionDeficit",
                     "S", "U", "X", "Y"):
            assert name not in params, f"structural label {name!r} leaked into free params"


class TestExpressionWalker:
    """Tests for the SymPy-to-JAX expression walker."""

    def test_arithmetic(self):
        """Basic arithmetic expressions are correctly converted."""
        from flowprog.compilers.numpyro import _walk

        processes, objects = _make_recycling_structure()
        builder = ModelBuilder(processes, objects)

        a, b = sy.symbols("a b")
        env = {a: jnp.float64(3.0), b: jnp.float64(4.0)}

        # a + b
        result = float(_walk(a + b, env, builder.structure, {}))
        assert abs(result - 7.0) < 1e-10

        # a * b
        result = float(_walk(a * b, env, builder.structure, {}))
        assert abs(result - 12.0) < 1e-10

        # a / b
        result = float(_walk(a / b, env, builder.structure, {}))
        assert abs(result - 0.75) < 1e-10

        # a ** 2
        result = float(_walk(a ** 2, env, builder.structure, {}))
        assert abs(result - 9.0) < 1e-10

    def test_constant_lookup(self):
        """Constants (recipe values) are correctly resolved from the constants dict."""
        from flowprog.compilers.numpyro import _walk

        processes, objects = _make_recycling_structure()
        builder = ModelBuilder(processes, objects)
        constants = _normalize_recipe(builder.structure, RECIPE_DATA)

        S = builder.structure.S
        result = float(_walk(S[0, 0], {}, builder.structure, constants))
        assert abs(result - 1.0) < 1e-10

        U = builder.structure.U
        result = float(_walk(U[1, 0], {}, builder.structure, constants))
        assert abs(result - 0.8) < 1e-10

    def test_env_takes_priority_over_constants(self):
        """When the same symbol is in both env and constants, env wins."""
        from flowprog.compilers.numpyro import _walk

        processes, objects = _make_recycling_structure()
        builder = ModelBuilder(processes, objects)

        S = builder.structure.S
        constants = {S[0, 0]: 1.0}
        env = {S[0, 0]: jnp.float64(42.0)}

        result = float(_walk(S[0, 0], env, builder.structure, constants))
        assert abs(result - 42.0) < 1e-10

    def test_max_to_softplus(self):
        """Max(0, x) is converted to softplus(x)."""
        from flowprog.compilers.numpyro import _walk

        processes, objects = _make_recycling_structure()
        builder = ModelBuilder(processes, objects)

        x = sy.Symbol("x")
        env = {x: jnp.float64(5.0)}

        expr = sy.Max(0, x, evaluate=False)
        result = float(_walk(expr, env, builder.structure, {}))
        assert abs(result - 5.0) < 0.1  # softplus(5) ≈ 5

        env = {x: jnp.float64(-5.0)}
        result = float(_walk(expr, env, builder.structure, {}))
        assert abs(result - 0.0) < 0.1  # softplus(-5) ≈ 0


class TestNumpyroCompilerBasic:
    """Basic compilation tests for the NumPyro compiler."""

    def test_simple_model(self):
        """A simple one-step model compiles and evaluates correctly."""
        processes = [Process("M1", produces=["out"], consumes=["in"])]
        objects = [MObject("in"), MObject("out")]
        builder = ModelBuilder(processes, objects)

        f = sy.Symbol("f", positive=True)
        builder.add({builder.X[0]: f}, label="test")

        recipe = {"M1": {"produces": {"out": 1.0}, "consumes": {"in": 1.0}}}
        model_fn = compile_numpyro(
            builder.structure,
            builder._steps,
            recipe_data=recipe,
            beta=1000.0,
        )

        # Pass f as keyword argument
        with handlers.seed(rng_seed=0):
            trace = handlers.trace(model_fn).get_trace(f=3.5)

        assert abs(float(trace["X_0_M1"]["value"]) - 3.5) < 1e-6

    def test_compiles_with_spec_priors(self):
        """Parameters with priors in spec are sampled."""
        processes = [Process("M1", produces=["out"], consumes=["in"])]
        objects = [MObject("in"), MObject("out")]
        builder = ModelBuilder(processes, objects)

        f = sy.Symbol("f", positive=True)
        builder.add({builder.X[0]: f}, label="test")

        spec = ModelSpec(priors={"f": dist.Normal(3.5, 0.1)})
        recipe = {"M1": {"produces": {"out": 1.0}, "consumes": {"in": 1.0}}}
        model_fn = compile_numpyro(
            builder.structure,
            builder._steps,
            spec=spec,
            recipe_data=recipe,
        )

        # Should run without error (f is sampled, not provided)
        with handlers.seed(rng_seed=42):
            trace = handlers.trace(model_fn).get_trace()

        assert "f" in trace
        assert trace["f"]["type"] == "sample"

    def test_spec_priors_accept_symbol_keys(self):
        """ModelSpec.priors accepts sympy Symbol keys in addition to strings."""
        processes = [Process("M1", produces=["out"], consumes=["in"])]
        objects = [MObject("in"), MObject("out")]
        builder = ModelBuilder(processes, objects)

        f = sy.Symbol("f", positive=True)
        builder.add({builder.X[0]: f}, label="test")

        spec = ModelSpec(priors={f: dist.Normal(3.5, 0.1)})
        recipe = {"M1": {"produces": {"out": 1.0}, "consumes": {"in": 1.0}}}
        model_fn = compile_numpyro(
            builder.structure,
            builder._steps,
            spec=spec,
            recipe_data=recipe,
        )

        with handlers.seed(rng_seed=42):
            trace = handlers.trace(model_fn).get_trace()

        assert "f" in trace
        assert trace["f"]["type"] == "sample"


class TestRecyclingModelComparison:
    """Compare NumPyro compiler output against SymPy compiler for the
    three-step recycling model.

    For parameter values away from the regime boundary, results should
    agree to numerical precision (modulo smooth approximation error which
    is O(1/beta)).
    """

    def test_f5_g1_no_deficit(self):
        """f=5, g=1: small supply, large deficit → P3 fills the gap.

        Demand for O2 from P1: 5 * 0.8 / 1.0 = 4.0
        Supply from P2: 1 * 1.0 / 1.0 = 1.0
        Deficit: 4.0 - 1.0 = 3.0
        P3 activity: 3.0 / 1.0 = 3.0
        """
        builder, f, g = _build_recycling_model(5, 1)

        sympy_results = _eval_sympy_model(builder, 5, 1)
        numpyro_results = _eval_numpyro_model(builder, 5, 1, beta=1000.0)

        for key in ["X_0", "Y_0", "X_1", "Y_1", "X_2", "Y_2"]:
            assert abs(sympy_results[key] - numpyro_results[key]) < 0.05, (
                f"{key}: sympy={sympy_results[key]:.4f}, numpyro={numpyro_results[key]:.4f}"
            )

        # Check specific expected values
        assert abs(numpyro_results["Y_0"] - 5.0) < 0.05  # P1 activity
        assert abs(numpyro_results["Y_1"] - 1.0) < 0.05  # P2 activity
        assert abs(numpyro_results["Y_2"] - 3.0) < 0.05  # P3 activity

    def test_f5_g4_near_boundary(self):
        """f=5, g=4: supply nearly meets demand.

        Demand for O2: 5 * 0.8 = 4.0
        Supply from P2: 4 * 1.0 = 4.0
        Deficit: 0.0
        P3 activity: 0.0
        """
        builder, f, g = _build_recycling_model(5, 4)

        sympy_results = _eval_sympy_model(builder, 5, 4)
        numpyro_results = _eval_numpyro_model(builder, 5, 4, beta=1000.0)

        # P1 and P2 should be close
        assert abs(numpyro_results["Y_0"] - 5.0) < 0.05
        assert abs(numpyro_results["Y_1"] - 4.0) < 0.05

        # P3 should be near zero (within smooth approximation tolerance)
        assert abs(numpyro_results["Y_2"]) < 0.05

    def test_f5_g6_excess_supply(self):
        """f=5, g=6: supply exceeds demand → P3 = 0.

        Demand for O2: 5 * 0.8 = 4.0
        Supply from P2: 6 * 1.0 = 6.0
        Deficit: max(0, 4.0 - 6.0) = 0
        P3 activity: 0
        """
        builder, f, g = _build_recycling_model(5, 6)

        sympy_results = _eval_sympy_model(builder, 5, 6)
        numpyro_results = _eval_numpyro_model(builder, 5, 6, beta=1000.0)

        assert abs(numpyro_results["Y_0"] - 5.0) < 0.05
        assert abs(numpyro_results["Y_1"] - 6.0) < 0.05
        assert abs(numpyro_results["Y_2"]) < 0.05  # Should be ~0

    def test_smoothness_bounded_deviation(self):
        """Near regime boundary, smooth deviation is bounded by O(1/beta)."""
        builder, f, g = _build_recycling_model(5, 4)

        # With beta=100, deviation should be O(0.01)
        results_100 = _eval_numpyro_model(builder, 5, 4, beta=100.0)
        # With beta=1000, deviation should be O(0.001)
        results_1000 = _eval_numpyro_model(builder, 5, 4, beta=1000.0)

        # The higher-beta result should be closer to the exact answer
        sympy_results = _eval_sympy_model(builder, 5, 4)

        err_100 = abs(results_100["Y_2"] - sympy_results["Y_2"])
        err_1000 = abs(results_1000["Y_2"] - sympy_results["Y_2"])

        # Higher beta → smaller error
        assert err_1000 < err_100 or (err_100 < 0.01 and err_1000 < 0.01)


class TestRecyclingModelWithLimit:
    """Test the recycling model variant where step 2 has a limit:
    P2's contribution to O2 cannot exceed the demand from step 1.
    """

    def _build_limited_model(self):
        """Build the recycling model with a limit on P2's O2 production."""
        processes, objects = _make_recycling_structure()
        builder = ModelBuilder(processes, objects)

        f = sy.Symbol("f", positive=True)
        g = sy.Symbol("g", positive=True)

        # Step 1: Pull production of O1 = f, stopping at O2
        step1 = builder.pull_production("O1", f, until_objects={"O2"})
        builder.add(step1, label="demand")

        # Step 2: Push consumption of O4 = g, stopping at O2
        # WITH LIMIT: O2 supply from P2 ≤ O2 demand from step 1
        step2 = builder.push_consumption("O4", g, until_objects={"O2"})

        # The expression to limit is the total production of O2
        # and the limit is the total consumption of O2
        limit_expr = builder.expr("SoldProduction", object_id="O2")
        demand_expr = builder.expr("Consumption", object_id="O2")
        step2 = builder.limit(step2, limit_expr, demand_expr)
        builder.add(step2, label="recyclate supply (limited)")

        # Step 3: Pull production deficit of O2 from P3 specifically
        deficit = builder.object_production_deficit("O2")
        step3 = builder.pull_process_output("P3", "O2", deficit, until_objects={"O2"})
        builder.add(step3, label="deficit production")

        return builder, f, g

    def test_limit_g6_caps_supply(self):
        """f=5, g=6 with limit: P2's O2 supply capped at O2 demand (4.0).

        Without limit: P2 supplies 6.0 of O2
        With limit: P2 supplies min(6.0, 4.0) = 4.0 of O2
        """
        builder, f, g = self._build_limited_model()

        sympy_results = _eval_sympy_model(builder, 5, 6)
        numpyro_results = _eval_numpyro_model(builder, 5, 6, beta=1000.0)

        # P1 activity should be f / S₁₁ = 5
        assert abs(numpyro_results["Y_0"] - 5.0) < 0.1

        # P2 activity should be capped: g would give 6.0, but limit caps
        # the Y[1] to 4.0 (demand for O2 is f * U₂₁ / S₁₁ = 4.0)
        # So Y[1] = 4.0
        assert abs(numpyro_results["Y_1"] - sympy_results["Y_1"]) < 0.1

        # P3 should be near zero (supply meets demand)
        assert abs(numpyro_results["Y_2"] - sympy_results["Y_2"]) < 0.1

    def test_limit_g1_no_cap(self):
        """f=5, g=1 with limit: supply (1.0) < demand (4.0) → no cap applies."""
        builder, f, g = self._build_limited_model()

        sympy_results = _eval_sympy_model(builder, 5, 1)
        numpyro_results = _eval_numpyro_model(builder, 5, 1, beta=1000.0)

        for key in ["X_0", "Y_0", "X_1", "Y_1", "X_2", "Y_2"]:
            assert abs(sympy_results[key] - numpyro_results[key]) < 0.1, (
                f"{key}: sympy={sympy_results[key]:.4f}, numpyro={numpyro_results[key]:.4f}"
            )


class TestObservations:
    """Tests for observation handling in the NumPyro compiler."""

    def test_observation_emitted(self):
        """Observations produce numpyro.sample sites in the trace."""
        processes = [Process("M1", produces=["out"], consumes=["in"])]
        objects = [MObject("in"), MObject("out")]
        builder = ModelBuilder(processes, objects)

        f = sy.Symbol("f", positive=True)
        builder.add({builder.Y[0]: f}, label="test")

        obs = Observation(
            name="total_out",
            expression=builder.structure.S[1, 0] * builder.structure.Y[0],
            noise=dist.Normal,
            sigma=1.0,
        )

        recipe = {"M1": {"produces": {"out": 1.0}, "consumes": {"in": 1.0}}}
        model_fn = compile_numpyro(
            builder.structure,
            builder._steps,
            recipe_data=recipe,
            observations=[obs],
        )

        with handlers.seed(rng_seed=0):
            trace = handlers.trace(model_fn).get_trace(
                obs={"total_out": jnp.array([5.0])}, f=5.0
            )

        assert "obs_total_out" in trace

    def test_conflicting_observations_posterior_between(self):
        """Two conflicting observations pull the posterior between their values.

        Model: Y[0] = f, observed as S[1,0]*Y[0] with two data points.
        obs_low says the value is 3.0 (sigma=0.1).
        obs_high says the value is 7.0 (sigma=0.1).
        The posterior mean of f should be approximately centred at 5.0.
        """
        processes = [Process("M1", produces=["out"], consumes=["in"])]
        objects = [MObject("in"), MObject("out")]
        builder = ModelBuilder(processes, objects)

        f = sy.Symbol("f", positive=True)
        builder.add({builder.Y[0]: f}, label="test")

        expr = builder.structure.S[1, 0] * builder.structure.Y[0]
        obs_low = Observation(
            name="obs_low",
            expression=expr,
            noise=dist.Normal,
            sigma=0.1,
        )
        obs_high = Observation(
            name="obs_high",
            expression=expr,
            noise=dist.Normal,
            sigma=0.1,
        )

        spec = ModelSpec(priors={"f": dist.Normal(5.0, 10.0)})
        recipe = {"M1": {"produces": {"out": 1.0}, "consumes": {"in": 1.0}}}
        model_fn = compile_numpyro(
            builder.structure,
            builder._steps,
            spec=spec,
            recipe_data=recipe,
            observations=[obs_low, obs_high],
        )

        # Run MCMC to get posterior samples
        kernel = numpyro.infer.NUTS(model_fn)
        mcmc = numpyro.infer.MCMC(kernel, num_warmup=200, num_samples=500, progress_bar=False)
        mcmc.run(
            jax.random.PRNGKey(0),
            obs={"obs_low": jnp.array(3.0), "obs_high": jnp.array(7.0)},
        )

        samples = mcmc.get_samples()
        posterior_mean = float(jnp.mean(samples["f"]))

        # The posterior mean should be approximately centred between 3.0 and 7.0
        assert abs(posterior_mean - 5.0) < 0.5, (
            f"Posterior mean {posterior_mean:.2f} should be near 5.0 "
            f"(midpoint of conflicting observations 3.0 and 7.0)"
        )
