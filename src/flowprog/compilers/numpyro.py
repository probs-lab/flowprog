"""NumPyro compiler: compiles model steps into a NumPyro model function.

Mirrors the structure of compile_sympy but produces a JAX-compatible
callable with numpyro.sample() and numpyro.deterministic() calls instead
of accumulated SymPy expressions.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import sympy as sy

from ..activities import Limit
from .numpyro_utils import smooth_clamp, softplus


@dataclass
class Observation:
    """Specification for an observed quantity.

    :param name: Identifier for this observation (used in numpyro.sample name).
    :param expression: Sympy expression over model symbols (X[j], Y[j], S[i,j], etc.)
        that computes the predicted value from the model state.
    :param noise: NumPyro distribution family for the observation noise (e.g. dist.Normal).
    :param sigma: Noise scale — either a float (fixed) or a string (name of a parameter
        to be sampled, which must be provided in the spec).
    """

    name: str
    expression: sy.Expr
    noise: object = dist.Normal  # distribution class
    sigma: object = 1.0  # float or str


@dataclass
class ModelSpec:
    """Specification of priors for model parameters.

    Maps parameter names (str or sympy Symbol) to their prior distributions
    (numpyro.distributions instances).
    """

    priors: dict = field(default_factory=dict)

    def _lookup(self, param_name: str):
        """Look up a prior by string name, accepting both str and Symbol keys."""
        if param_name in self.priors:
            return self.priors[param_name]
        # Try matching Symbol keys by name
        for key, value in self.priors.items():
            if isinstance(key, sy.Basic) and getattr(key, "name", None) == param_name:
                return value
        return None


def _walk(expr, env, structure, constants, beta=100.0):
    """Recursively walk a SymPy expression tree, producing JAX values.

    :param expr: SymPy expression to convert.
    :param env: dict mapping SymPy symbols/Indexed to JAX values (parameters,
        intermediates, accumulated X[j]/Y[j]).
    :param structure: ModelStructure for identifying structural symbols.
    :param constants: dict mapping SymPy Indexed symbols (e.g. S[i, j], U[i, j])
        to float values. Any sympy Indexed present in both env and constants will
        use the env value (env takes priority).
    :param beta: smoothness parameter for smooth approximations.
    :returns: JAX scalar value.
    """

    # --- Numeric literals ---
    if isinstance(expr, (sy.Integer, sy.Float, sy.Rational)):
        return jnp.float64(float(expr))

    if isinstance(expr, sy.NumberSymbol):
        return jnp.float64(float(expr))

    if expr is sy.S.Zero:
        return jnp.float64(0.0)

    if expr is sy.S.One:
        return jnp.float64(1.0)

    if expr is sy.S.NegativeOne:
        return jnp.float64(-1.0)

    if expr is sy.S.Half:
        return jnp.float64(0.5)

    # --- Indexed symbols ---
    if isinstance(expr, sy.Indexed):
        # env takes priority (accumulated state, sampled parameters)
        if expr in env:
            return env[expr]

        # Structural symbols: Balance, ProductionDeficit, ConsumptionDeficit
        base = expr.base
        if base == structure.Balance:
            idx = int(expr.indices[0])
            return _compute_balance_jax(structure, structure.objects[idx].id,
                                        env, constants)
        elif base == structure.ProductionDeficit:
            idx = int(expr.indices[0])
            balance = _compute_balance_jax(structure, structure.objects[idx].id,
                                           env, constants)
            return softplus(-balance, beta)
        elif base == structure.ConsumptionDeficit:
            idx = int(expr.indices[0])
            balance = _compute_balance_jax(structure, structure.objects[idx].id,
                                           env, constants)
            return softplus(balance, beta)

        # Constants (recipe coefficients or any user-supplied constants)
        if expr in constants:
            return jnp.float64(constants[expr])

        # X[j] and Y[j] not yet accumulated default to 0
        # (mirrors sympy compiler: values.get(sym, sy.S.Zero))
        if base == structure.X or base == structure.Y:
            return jnp.float64(0.0)

        raise KeyError(f"Unresolved indexed symbol: {expr}")

    # --- Plain symbols (parameters, intermediates) ---
    if isinstance(expr, sy.Symbol):
        if expr in env:
            return env[expr]
        raise KeyError(f"Unresolved symbol: {expr}")

    # --- Max(a, b) → smooth_max or softplus ---
    if isinstance(expr, sy.Max):
        args = expr.args
        if len(args) == 2:
            a = _walk(args[0], env, structure, constants, beta)
            b = _walk(args[1], env, structure, constants, beta)
            if args[0] == sy.S.Zero:
                return softplus(b, beta)
            if args[1] == sy.S.Zero:
                return softplus(a, beta)
            from .numpyro_utils import smooth_max
            return smooth_max(a, b, beta)
        from .numpyro_utils import smooth_max
        result = _walk(args[0], env, structure, constants, beta)
        for arg in args[1:]:
            result = smooth_max(result, _walk(arg, env, structure, constants, beta), beta)
        return result

    # --- Min(a, b) ---
    if isinstance(expr, sy.Min):
        args = expr.args
        from .numpyro_utils import smooth_max
        result = _walk(args[0], env, structure, constants, beta)
        for arg in args[1:]:
            b = _walk(arg, env, structure, constants, beta)
            result = -smooth_max(-result, -b, beta)
        return result

    # --- Piecewise ---
    if isinstance(expr, sy.Piecewise):
        raise ValueError(
            "Piecewise expressions should not appear in the NumPyro compiler. "
            "Limits should be handled via smooth_clamp."
        )

    # --- Arithmetic: Add ---
    if isinstance(expr, sy.Add):
        terms = [_walk(arg, env, structure, constants, beta) for arg in expr.args]
        result = terms[0]
        for t in terms[1:]:
            result = result + t
        return result

    # --- Arithmetic: Mul ---
    if isinstance(expr, sy.Mul):
        factors = [_walk(arg, env, structure, constants, beta) for arg in expr.args]
        result = factors[0]
        for f in factors[1:]:
            result = result * f
        return result

    # --- Arithmetic: Pow ---
    if isinstance(expr, sy.Pow):
        base_val = _walk(expr.args[0], env, structure, constants, beta)
        exp_val = _walk(expr.args[1], env, structure, constants, beta)
        return jnp.power(base_val, exp_val)

    # --- Abs ---
    if isinstance(expr, sy.Abs):
        return jnp.abs(_walk(expr.args[0], env, structure, constants, beta))

    # --- Fallback: try to convert via float ---
    try:
        return jnp.float64(float(expr))
    except (TypeError, ValueError):
        pass

    raise NotImplementedError(
        f"Cannot convert SymPy expression to JAX: {expr} (type: {type(expr).__name__})"
    )


def _compute_balance_jax(structure, object_id, env, constants):
    """Compute (production - consumption) for an object as a JAX value.

    :param structure: ModelStructure
    :param object_id: Object identifier string
    :param env: dict mapping SymPy Indexed (X[j], Y[j]) to JAX values
    :param constants: dict mapping SymPy Indexed (S[i,j], U[i,j]) to floats
    :returns: JAX scalar
    """
    i = structure.lookup_object(object_id)
    flow_in = jnp.float64(0.0)
    for j in structure._processes_producing_object.get(i, []):
        s_sym = structure.S[i, j]
        s_val = env.get(s_sym, jnp.float64(constants.get(s_sym, 0.0)))
        y_val = env.get(structure.Y[j], jnp.float64(0.0))
        flow_in = flow_in + s_val * y_val

    flow_out = jnp.float64(0.0)
    for j in structure._processes_consuming_object.get(i, []):
        u_sym = structure.U[i, j]
        u_val = env.get(u_sym, jnp.float64(constants.get(u_sym, 0.0)))
        x_val = env.get(structure.X[j], jnp.float64(0.0))
        flow_out = flow_out + u_val * x_val

    return flow_in - flow_out


def _compile_limit_jax(values_dict, limit_transform, env, structure, constants, beta=100.0):
    """Compile a Limit transformation using smooth approximation.

    Mirrors _compile_limit from the sympy compiler but uses smooth_clamp
    instead of Piecewise.

    :param values_dict: dict mapping SymPy symbols (X[j], Y[j]) to JAX values
        (the proposed additions from this step)
    :param limit_transform: Limit instance
    :param env: dict mapping SymPy symbols to JAX values (accumulated + intermediates)
    :param structure: ModelStructure
    :param constants: dict mapping SymPy Indexed to float values
    :param beta: smoothness parameter
    :returns: new values_dict with limited JAX values
    """
    # 'current' = expression evaluated with current env
    current = _walk(limit_transform.expression, env, structure, constants, beta)

    # 'limit' = limit_value evaluated with current env
    limit_val = _walk(limit_transform.limit_value, env, structure, constants, beta)

    # 'proposed' = expression evaluated with (env + step contribution)
    proposed_env = dict(env)
    for j in range(len(structure.processes)):
        xj = structure.X[j]
        yj = structure.Y[j]
        if xj in values_dict:
            proposed_env[xj] = env.get(xj, jnp.float64(0.0)) + values_dict[xj]
        if yj in values_dict:
            proposed_env[yj] = env.get(yj, jnp.float64(0.0)) + values_dict[yj]

    proposed = _walk(limit_transform.expression, proposed_env, structure, constants, beta)

    diff = proposed - current
    safe_diff = jnp.maximum(diff, 1e-10)
    raw_fraction = (limit_val - current) / safe_diff
    fraction = smooth_clamp(raw_fraction, 0.0, 1.0, beta)

    return {k: fraction * v for k, v in values_dict.items()}


def compile_numpyro(
    structure,
    steps,
    spec=None,
    recipe_data=None,
    observations=None,
    beta=100.0,
):
    """Compile model steps into a NumPyro model function.

    Mirrors compile_sympy but produces a callable NumPyro model function
    instead of accumulated SymPy expressions.

    :param structure: ModelStructure (processes, objects, connectivity)
    :param steps: list[AdditionalActivity] — the recorded model-building steps
    :param spec: ModelSpec — prior specifications for parameters (optional;
        if None, all free parameters must be passed as keyword arguments)
    :param recipe_data: dict — recipe values in one of two formats:
        - ID-based: {process_id: {consumes: {obj_id: val}, produces: {obj_id: val}}}
        - Sympy-indexed: {S[i, j]: val, U[i, j]: val}
    :param observations: list[Observation] — observed data specifications
    :param beta: float — smoothness parameter for smooth approximations
    :returns: callable NumPyro model function
    """
    if spec is None:
        spec = ModelSpec()
    if observations is None:
        observations = []

    # Convert recipe_data to sympy-indexed format
    constants = _normalize_recipe(structure, recipe_data)

    # Analyze steps to find free parameters
    free_params = _collect_free_parameters(structure, steps)

    def model(obs=None, **params):
        # Maps sympy symbols/indexed to JAX values
        env = {}

        # Sample or look up each free parameter
        for param_name, param_sym in free_params.items():
            if param_name in params:
                env[param_sym] = jnp.asarray(params[param_name], dtype=jnp.float64)
            elif (prior_dist := spec._lookup(param_name)) is not None:
                val = numpyro.sample(param_name, prior_dist)
                env[param_sym] = val
            else:
                raise ValueError(f"Parameter '{param_name}' must be assigned a prior or passed as kwarg")

        # Accumulated state: maps X[j], Y[j] to JAX values
        acc = {}

        # Walk steps, mirroring compile_sympy
        for step in steps:
            # 1. Resolve intermediates
            for sym, expr, desc in step.intermediates:
                resolved = _walk(expr, {**acc, **env}, structure, constants, beta)
                env[sym] = resolved

            # 2. Resolve step values to JAX
            resolved_values = {}
            for sym, expr in step.values.items():
                resolved_values[sym] = _walk(
                    expr, {**acc, **env}, structure, constants, beta
                )

            # 3. Apply transformations
            for transform in step.transformations:
                if isinstance(transform, Limit):
                    resolved_values = _compile_limit_jax(
                        resolved_values, transform,
                        {**acc, **env}, structure, constants, beta
                    )
                else:
                    raise ValueError(f"Unknown transform {transform}")

            # 4. Accumulate
            for sym, val in resolved_values.items():
                if sym in acc:
                    acc[sym] = acc[sym] + val
                else:
                    acc[sym] = val

        # Track deterministic quantities for all process activities
        for j, proc in enumerate(structure.processes):
            xj = structure.X[j]
            yj = structure.Y[j]
            if xj in acc:
                numpyro.deterministic(f"X_{j}_{proc.id}", acc[xj])
            if yj in acc:
                numpyro.deterministic(f"Y_{j}_{proc.id}", acc[yj])

        # Emit observations
        all_env = {**acc, **env}
        for observation in observations:
            predicted = _walk(
                observation.expression, all_env, structure, constants, beta
            )

            # Resolve sigma
            if isinstance(observation.sigma, str):
                sigma_name = observation.sigma
                sigma_symbol = sy.Symbol(sigma_name)
                if sigma_symbol in env:
                    sigma_val = env[sigma_symbol]
                elif sigma_name in params:
                    sigma_val = jnp.float64(params[sigma_name])
                elif (prior_dist := spec._lookup(sigma_name)) is not None:
                    sigma_val = numpyro.sample(sigma_name, prior_dist)
                    env[sigma_symbol] = sigma_val
                else:
                    raise ValueError(f"Unknown sigma '{sigma_name}'")
            else:
                sigma_val = jnp.float64(observation.sigma)

            obs_data = None
            if obs is not None and observation.name in obs:
                obs_data = obs[observation.name]
                numpyro.sample(
                    f"obs_{observation.name}",
                    observation.noise(predicted, sigma_val),
                    obs=obs_data,
                )

            # # Use plate for multiple observations (1-d array with len > 1)
            # obs_ndim = getattr(obs_data, 'ndim', 0) if obs_data is not None else 0
            # if obs_ndim >= 1 and obs_data.shape[0] > 1:
            #     with numpyro.plate(f"obs_{observation.name}_plate", obs_data.shape[0]):
            #         numpyro.sample(
            #             f"obs_{observation.name}",
            #             observation.noise(predicted, sigma_val),
            #             obs=obs_data,
            #         )
            # else:
            #     # Scalar or single-element observation
            #     scalar_obs = obs_data
            #     if scalar_obs is not None and obs_ndim >= 1:
            #         scalar_obs = obs_data[0]
            #     numpyro.sample(
            #         f"obs_{observation.name}",
            #         observation.noise(predicted, sigma_val),
            #         obs=scalar_obs,
            #     )

    return model


def _normalize_recipe(structure, recipe_data):
    """Convert recipe_data to a dict mapping SymPy Indexed symbols to float values.

    Accepts either:
    - ID-based: {process_id: {consumes: {obj_id: val}, produces: {obj_id: val}}}
    - SymPy-indexed: {S[i, j]: val, U[i, j]: val}
    - None: returns empty dict
    """
    if recipe_data is None:
        return {}

    if not recipe_data:
        return {}

    # Check if already in sympy-indexed format
    first_key = next(iter(recipe_data))
    if isinstance(first_key, sy.Indexed):
        return dict(recipe_data)

    # Convert from ID-based format to sympy-indexed
    result = {}
    for proc_id, flows in recipe_data.items():
        j = structure.lookup_process(proc_id)
        if "produces" in flows:
            for obj_id, val in flows["produces"].items():
                i = structure.lookup_object(obj_id)
                result[structure.S[i, j]] = float(val)
        if "consumes" in flows:
            for obj_id, val in flows["consumes"].items():
                i = structure.lookup_object(obj_id)
                result[structure.U[i, j]] = float(val)

    return result


def _collect_free_parameters(structure, steps):
    """Scan steps to identify free parameters (symbols not part of model structure).

    Returns a dict mapping parameter name (str) to sympy Symbol.
    """
    model_bases = {structure.X, structure.Y, structure.S, structure.U,
                   structure.Balance, structure.ProductionDeficit,
                   structure.ConsumptionDeficit}

    intermediate_syms = set()
    for step in steps:
        for sym, _, _ in step.intermediates:
            intermediate_syms.add(sym)

    free_params = {}
    for step in steps:
        for _, expr, _ in step.intermediates:
            _extract_free_from_expr(expr, model_bases, intermediate_syms, free_params)
        for _, expr in step.values.items():
            _extract_free_from_expr(expr, model_bases, intermediate_syms, free_params)
        for transform in step.transformations:
            if isinstance(transform, Limit):
                _extract_free_from_expr(
                    transform.limit_value, model_bases, intermediate_syms, free_params
                )

    return free_params


def _extract_free_from_expr(expr, model_bases, intermediate_syms, result):
    """Extract free symbols from a sympy expression, excluding model symbols."""
    if not isinstance(expr, sy.Basic):
        return

    # Collect labels of model IndexedBases so we can exclude them from
    # free_symbols (sympy exposes IndexedBase labels as plain Symbols).
    model_base_labels = {b.label for b in model_bases}

    for sym in expr.free_symbols:
        if sym in intermediate_syms:
            continue
        if isinstance(sym, sy.Indexed):
            continue
        if sym in model_base_labels:
            continue
        result[sym.name] = sym

    for indexed in expr.atoms(sy.Indexed):
        if indexed.base not in model_bases:
            result[str(indexed)] = indexed
