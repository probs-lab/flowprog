"""NumPyro compiler: compiles model steps into a NumPyro model function.

Mirrors the structure of compile_sympy but produces a JAX-compatible
callable with numpyro.sample() and numpyro.deterministic() calls instead
of accumulated SymPy expressions.
"""

from dataclasses import dataclass, replace

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import sympy as sy

from ...activities import Limit, Floor
from .sympy_to_jax import sympy_expr_to_jax
from .transform_handlers import assign_transform_handlers, default_transform_handler


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


class NumpyroState:
    def __init__(self, structure, constants, initial_env, beta=100.0):
        self.structure = structure
        self.constants = constants
        self.beta = beta
        # TODO should these really all be separate dicts for constants,
        # accumulated X/Y, and other env parameters?
        self.accumulated = {}  # {X[j]: jax_value, Y[j]: jax_value}
        self._jax_values = dict(initial_env)  # includes sampled params

    def eval(self, expr, with_proposed=None):
        values = dict(self._jax_values)
        values.update(self.accumulated)
        if with_proposed:
            for k, v in with_proposed.items():
                values[k] = values.get(k, 0.0) + v
        return sympy_expr_to_jax(
            expr, values, self.structure, self.constants, self.beta
        )

    def resolve_intermediates(self, intermediates):
        """Process and store intermediate definitions."""
        for sym, expr, desc in intermediates:
            self.store_intermediate(sym, self.eval(expr), desc)

    def store_intermediate(self, sym, value, desc):
        self._jax_values[sym] = value

    def accumulate(self, proposed):
        for k, v in proposed.items():
            self.accumulated[k] = self.accumulated.get(k, 0.0) + v


class NumpyroModel:
    @classmethod
    def from_steps(
        cls,
        steps,
        structure,
        param_priors=None,
        recipe_data=None,
        observations=None,
        surplus_parameterisation=None,
        beta=100.0,
    ):
        """Compile flowprog steps ready for numpyro sampling.

        :param builder: ModelBuilder (with structure and steps attributes)
        :param recipe_data: dict — recipe values in one of two formats:
            - ID-based: {process_id: {consumes: {obj_id: val}, produces: {obj_id: val}}}
            - Sympy-indexed: {S[i, j]: val, U[i, j]: val}
        :param observations: list[Observation] — observed data specifications
        :param beta: float — smoothness parameter for smooth approximations
        """
        # Convert recipe_data to sympy-indexed format
        constants = _normalize_recipe(structure, recipe_data)

        # Ensure all steps' transformations have names
        steps = _ensure_transformation_names(steps)

        # Determine transforms and reparameterisation
        if param_priors is None:
            param_priors = {}

        # Ensure sympy symbols are normalised to their names
        param_priors = {str(k): v for k, v in param_priors.items()}

        handlers, param_priors = assign_transform_handlers(
            steps, param_priors, surplus_parameterisation, beta
        )

        return cls(
            structure,
            steps,
            constants,
            param_priors,
            observations,
            handlers,
            beta,
        )

    def __init__(
        self,
        structure,
        steps,
        constant_data,
        param_priors,
        observations=None,
        transform_handlers=None,
        beta=100.0,
    ):
        if transform_handlers is None:
            transform_handlers = {}
        self.structure = structure
        self.steps = steps
        self.constant_data = constant_data
        self.param_priors = param_priors
        self.observations = observations
        self.transform_handlers = transform_handlers
        self.beta = beta

    def __call__(self, params, handlers=None):
        """Run forward model. Returns result dict."""

        # TODO should decide whether params is keyed by string or symbol
        # Analyze steps to find free parameters
        free_params = _collect_free_parameters(self.structure, self.steps)
        env = {}
        for param_name, param_sym in free_params.items():
            if param_name in params:
                env[param_sym] = jnp.asarray(params[param_name], dtype=jnp.float64)
            elif param_sym in params:
                env[param_sym] = jnp.asarray(params[param_sym], dtype=jnp.float64)
            # TODO check if this is right - currently disabled since when using
            # reparameterisation this can be a false positive, e.g. the capacity
            # does not appear in params but that's because it's sampled later by
            # the transform.
            #
            # else:
            #     raise ValueError(
            #         f"Parameter '{param_name}' must be assigned a prior or passed as kwarg"
            #     )

        h = {**self.transform_handlers, **(handlers or {})}
        state = NumpyroState(self.structure, self.constant_data, env, self.beta)

        for step in self.steps:
            state.resolve_intermediates(step.intermediates)
            proposed = {k: state.eval(v) for k, v in step.values.items()}
            for transform in step.transformations:
                if transform.name is not None and transform.name in h:
                    handler = h[transform.name]
                else:
                    handler = default_transform_handler(transform)(beta=self.beta)
                proposed = handler(proposed, state, transform)
            state.accumulate(proposed)

        return state.accumulated

    def store_process_activities(self, results):
        """Emit numpyro.deterministic for each process X[j] and Y[j].
        Wrap this in numpyro.plate if needed."""
        for j in range(len(self.structure.processes)):
            xj = self.structure.X[j]
            yj = self.structure.Y[j]
            if xj in results:
                numpyro.deterministic(f"X_{j}", results[xj])
            if yj in results:
                numpyro.deterministic(f"Y_{j}", results[yj])

    def sample_params(self):
        """Emit numpyro.sample for each parameter. Returns params dict."""
        if self.param_priors is None:
            raise ValueError("No parameter priors specified")
        return {
            name: numpyro.sample(name, prior)
            for name, prior in self.param_priors.items()
        }

    def likelihood(self, result, obs, params=None):
        """Emit numpyro.sample(obs=...) for each observation."""

        if self.observations is None:
            return

        if params is None:
            params = {}

        for o in self.observations:
            predicted = sympy_expr_to_jax(
                o.expression, result, self.structure, self.constant_data, self.beta
            )

            # Resolve sigma
            if isinstance(o.sigma, str):
                sigma_name = o.sigma
                # sigma_symbol = sy.Symbol(sigma_name)
                # if sigma_symbol in params:
                #     sigma_val = params[sigma_symbol]
                if sigma_name in params:
                    sigma_val = params[sigma_name]
                # TODO could define sigma_priors on NumpyroModel and do this still
                # elif (prior_dist := spec._lookup(sigma_name)) is not None:
                #     sigma_val = numpyro.sample(sigma_name, prior_dist)
                #     env[sigma_symbol] = sigma_val
                else:
                    raise ValueError(f"Unknown sigma '{sigma_name}'")
            else:
                sigma_val = jnp.float64(o.sigma)
            obs_data = obs[o.name]
            numpyro.sample(f"obs_{o.name}", o.noise(predicted, sigma_val), obs=obs_data)

    def numpyro_model(self, obs=None):
        """Complete numpyro model: sample, forward, observe."""
        params = self.sample_params()
        result = self(params)

        # Define shared observation plate (size inferred from obs data).
        # This plate is reused for all sample sites that vary per data point:
        # currently just observation likelihoods, but available for
        # per-datapoint latent variables (e.g. surplus parameterisation).
        obs_plate = None
        if obs is not None:
            obs_sizes = {
                name: data.shape[0] if data.ndim > 0 else None
                for name, data in obs.items()
            }
            unique_sizes = set(obs_sizes.values())
            if len(unique_sizes) > 1:
                raise ValueError(
                    f"All observation arrays must have the same length, "
                    f"got: {obs_sizes}"
                )
            N_obs = unique_sizes.pop()
            if N_obs is not None:
                obs_plate = numpyro.plate("obs_plate", N_obs)

        if obs_plate is not None:
            with obs_plate:
                self.store_process_activities(result)
                self.likelihood(result, obs, params)
        else:
            self.store_process_activities(result)
            self.likelihood(result, obs, params)


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


def _ensure_transformation_names(steps):
    result = []
    seen_names = set()
    for step_idx, step in enumerate(steps):
        transformations = []
        for t_idx, transform in enumerate(step.transformations):
            name = transform.name
            if name is None:
                name = f"_step{step_idx}_transform{t_idx}"
                transform = replace(transform, name=name)
            if name in seen_names:
                raise ValueError(f"Duplicate transform name '{name}'")
            transformations.append(transform)
        result.append(replace(step, transformations=transformations))
    return result


def _collect_free_parameters(structure, steps):
    """Scan steps to identify free parameters (symbols not part of model structure).

    Returns a dict mapping parameter name (str) to sympy Symbol.
    """
    model_bases = {
        structure.X,
        structure.Y,
        structure.S,
        structure.U,
        structure.Balance,
        structure.ProductionDeficit,
        structure.ConsumptionDeficit,
    }

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
            # TODO do we need to do this explicitly?
            if isinstance(transform, Limit):
                _extract_free_from_expr(
                    transform.limit_value, model_bases, intermediate_syms, free_params
                )
            if isinstance(transform, Floor):
                _extract_free_from_expr(
                    transform.threshold, model_bases, intermediate_syms, free_params
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
