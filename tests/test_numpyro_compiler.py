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

import jax
import jax.numpy as jnp
import numpyro
import numpyro.handlers as handlers
import numpyro.distributions as dist
import numpyro.infer
import sympy as sy

from flowprog.model import ModelBuilder, Process, Object
from flowprog.backends.numpyro import CompiledModel, Observation, ModelSpec
from flowprog.backends.numpyro.transform_handlers import SurplusLimitHandler

from .model_strategies import MObject


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


def _eval_numpyro_model(builder, f_val, g_val, beta=1000.0, surplus=False):
    """Evaluate the numpyro-compiled model for given f and g.

    Passes f and g as keyword arguments to the model function.
    """
    compiled_model = CompiledModel.from_steps(
        builder._steps,
        builder.structure,
        recipe_data=RECIPE_DATA,
        beta=beta,
        surplus_parameterisation=surplus,
    )

    def fn():
        results = compiled_model(params=dict(f=f_val, g=g_val))
        compiled_model.store_process_activities(results)

    # Run the model with fixed parameter values (passed as kwargs)
    with handlers.seed(rng_seed=0):
        trace = handlers.trace(fn).get_trace()

    results = {
        k: float(trace[k]["value"])
        for k in trace
        if k.startswith("X_") or k.startswith("Y_")
    }
    return results


# ── Tests ────────────────────────────────────────────────────────────


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
        compiled_model = CompiledModel.from_steps(
            builder._steps,
            builder.structure,
            recipe_data=recipe,
            beta=1000.0,
        )

        def fn():
            results = compiled_model(params=dict(f=3.5))
            compiled_model.store_process_activities(results)

        # Pass f as keyword argument
        with handlers.seed(rng_seed=0):
            trace = handlers.trace(fn).get_trace()

        assert abs(float(trace["X_0"]["value"]) - 3.5) < 1e-6

    def test_compiles_with_spec_priors(self):
        """Parameters with priors in spec are sampled."""
        processes = [Process("M1", produces=["out"], consumes=["in"])]
        objects = [MObject("in"), MObject("out")]
        builder = ModelBuilder(processes, objects)

        f = sy.Symbol("f", positive=True)
        builder.add({builder.X[0]: f}, label="test")

        spec = ModelSpec(priors={"f": dist.Normal(3.5, 0.1)})
        recipe = {"M1": {"produces": {"out": 1.0}, "consumes": {"in": 1.0}}}
        compiled_model = CompiledModel.from_steps(
            builder._steps,
            builder.structure,
            spec=spec,
            recipe_data=recipe,
        )

        # Should run without error (f is sampled, not provided)
        with handlers.seed(rng_seed=42):
            trace = handlers.trace(compiled_model.numpyro_model).get_trace()

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
        compiled_model = CompiledModel.from_steps(
            builder._steps,
            builder.structure,
            spec=spec,
            recipe_data=recipe,
        )

        with handlers.seed(rng_seed=42):
            trace = handlers.trace(compiled_model.numpyro_model).get_trace()

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
            assert (
                abs(sympy_results[key] - numpyro_results[key]) < 0.05
            ), f"{key}: sympy={sympy_results[key]:.4f}, numpyro={numpyro_results[key]:.4f}"

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

        for key in ["X_0", "Y_0", "X_1", "Y_1", "X_2", "Y_2"]:
            assert (
                abs(sympy_results[key] - numpyro_results[key]) < 0.05
            ), f"{key}: sympy={sympy_results[key]:.4f}, numpyro={numpyro_results[key]:.4f}"

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

        for key in ["X_0", "Y_0", "X_1", "Y_1", "X_2", "Y_2"]:
            assert (
                abs(sympy_results[key] - numpyro_results[key]) < 0.05
            ), f"{key}: sympy={sympy_results[key]:.4f}, numpyro={numpyro_results[key]:.4f}"

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
            assert (
                abs(sympy_results[key] - numpyro_results[key]) < 0.1
            ), f"{key}: sympy={sympy_results[key]:.4f}, numpyro={numpyro_results[key]:.4f}"

    def test_surplus_parameterisation(self):
        """Note that here it doesn't make any difference as the capacity limit
        is based on the consumption in the model, not an uncertain capacity
        parameter.  That case is tested below.

        """
        builder, f, g = self._build_limited_model()

        numpyro_natural_results = _eval_numpyro_model(
            builder, 5, 6, beta=1000.0, surplus=False
        )
        numpyro_surplus_results = _eval_numpyro_model(
            builder, 5, 6, beta=1000.0, surplus=True
        )

        for key in ["X_0", "Y_0", "X_1", "Y_1", "X_2", "Y_2"]:
            assert (
                abs(numpyro_natural_results[key] - numpyro_surplus_results[key]) < 0.1
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
        compiled_model = CompiledModel.from_steps(
            builder._steps,
            builder.structure,
            recipe_data=recipe,
            observations=[obs],
        )

        with handlers.seed(rng_seed=0):
            trace = handlers.trace(compiled_model.likelihood).get_trace(
                obs={"total_out": jnp.array([5.0])}, result={}
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
        compiled_model = CompiledModel.from_steps(
            builder._steps,
            builder.structure,
            spec=spec,
            recipe_data=recipe,
            observations=[obs_low, obs_high],
        )

        # Run MCMC to get posterior samples
        kernel = numpyro.infer.NUTS(compiled_model.numpyro_model)
        mcmc = numpyro.infer.MCMC(
            kernel, num_warmup=200, num_samples=500, progress_bar=False
        )
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


class TestNumpyroCompilerSuplusParameterisation:
    """Test surplus parameterisation."""

    def test_simple_model_automatic_reparameterisation(self):
        processes = [Process("M1", produces=["out"], consumes=[])]
        objects = [MObject("out")]
        builder = ModelBuilder(processes, objects)

        f = sy.Symbol("f", positive=True)
        C = sy.Symbol("C", positive=True)
        proposed = {builder.Y[0]: f}
        step = builder.limit(proposed, builder.Y[0], C)
        builder.add(step)

        recipe = {"M1": {"produces": {"out": 1.0}, "consumes": {}}}

        # Without surplus parameterisation, both params have priors to sample
        compiled_model = CompiledModel.from_steps(
            builder._steps,
            builder.structure,
            param_priors={f: dist.Normal(8, 1), C: dist.Normal(12, 1)},
            surplus_parameterisation=False,
        )

        assert "f" in compiled_model.param_priors
        assert "C" in compiled_model.param_priors

        assert compiled_model.transform_handlers == {}
        # #isinstance(
        #     compiled_model.transform_handlers["_step0_transform0"], NaturalLimitHandler
        # )

        # With surplus parameterisation

        compiled_model2 = CompiledModel.from_steps(
            builder._steps,
            builder.structure,
            param_priors={f: dist.Normal(8, 1), C: dist.Normal(12, 1)},
            surplus_parameterisation=True,
        )

        assert "f" in compiled_model2.param_priors
        assert "C" not in compiled_model2.param_priors

        handler = compiled_model2.transform_handlers["_step0_transform0"]
        assert isinstance(handler, SurplusLimitHandler)

    def test_simple_model_surplus_handler_capacity_not_sampled(self):
        processes = [Process("M1", produces=["out"], consumes=[])]
        objects = [MObject("out")]
        builder = ModelBuilder(processes, objects)

        f = sy.Symbol("f", positive=True)
        C = sy.Symbol("C", positive=True)
        proposed = {builder.Y[0]: f}
        step = builder.limit(proposed, builder.Y[0], C)
        builder.add(step)

        recipe = {"M1": {"produces": {"out": 1.0}, "consumes": {}}}

        compiled_model = CompiledModel.from_steps(
            builder._steps,
            builder.structure,
            param_priors={f: dist.Normal(8, 1), C: dist.Normal(12, 1)},
            recipe_data=recipe,
            surplus_parameterisation=True,
        )

        # Pass f as keyword argument
        with handlers.seed(rng_seed=0):
            trace = handlers.trace(compiled_model.numpyro_model).get_trace()

        assert trace["delta_C"]["type"] == "sample"
        assert trace["C"]["type"] == "deterministic"

    def test_simple_model_surplus_handler_mcmc(self):
        processes = [Process("M1", produces=["out"], consumes=[])]
        objects = [MObject("out")]
        builder = ModelBuilder(processes, objects)

        f = sy.Symbol("f", positive=True)
        C = sy.Symbol("C", positive=True)
        proposed = {builder.Y[0]: f}
        step = builder.limit(proposed, builder.Y[0], C)
        builder.add(step)

        recipe = {"M1": {"produces": {"out": 1.0}, "consumes": {}}}

        compiled_model = CompiledModel.from_steps(
            builder._steps,
            builder.structure,
            param_priors={f: dist.Normal(8, 1), C: dist.Normal(12, 1)},
            recipe_data=recipe,
            surplus_parameterisation=True,
        )

        # Run MCMC to get posterior samples
        kernel = numpyro.infer.NUTS(compiled_model.numpyro_model)
        mcmc = numpyro.infer.MCMC(
            kernel, num_warmup=200, num_samples=500, progress_bar=False
        )
        mcmc.run(jax.random.PRNGKey(0))

        samples = mcmc.get_samples()
        f_posterior_mean = float(jnp.mean(samples["f"]))
        C_posterior_mean = float(jnp.mean(samples["C"]))

        assert abs(f_posterior_mean - 8.0) < 0.5
        assert abs(C_posterior_mean - 12.0) < 0.5
