"""Handlers for transforms (Limit, Floor, etc)."""

import jax.numpy as jnp
import jax.nn
import numpyro
import numpyro.distributions as dist

from ...activities import Limit, Floor
from .utils import smooth_clamp


def apply_limit_fraction(proposed, current, limit_val, proposed_val, beta):
    diff = proposed_val - current
    raw_fraction = (limit_val - current) / jnp.maximum(diff, 1e-10)
    fraction = smooth_clamp(raw_fraction, 0.0, 1.0, beta)
    return {k: fraction * v for k, v in proposed.items()}


class NaturalLimitHandler:
    def __init__(self, beta):
        self.beta = beta

    def __call__(self, proposed, state, transform):
        current = state.eval(transform.expression)
        proposed_val = state.eval(transform.expression, with_proposed=proposed)
        limit_val = state.eval(transform.limit_value)
        return apply_limit_fraction(
            proposed, current, limit_val, proposed_val, self.beta
        )


class NaturalFloorHandler:
    def __init__(self, beta=20.0):
        self.beta = beta

    def __call__(self, proposed, state, transform):
        proposed_val = state.eval(transform.expression, with_proposed=proposed)
        threshold = state.eval(transform.threshold)
        gate = jax.nn.sigmoid(self.beta * (proposed_val - threshold))
        return {k: gate * v for k, v in proposed.items()}


class SurplusLimitHandler:
    def __init__(self, name, prior, beta):
        self.name = name
        self.prior = prior
        self.beta = beta

    def __call__(self, proposed, state, transform):
        current = state.eval(transform.expression)
        proposed_val = state.eval(transform.expression, with_proposed=proposed)
        delta = numpyro.sample(
            f"delta_{self.name}",
            dist.Normal(self.prior.loc - current, self.prior.scale),
        )
        limit_val = numpyro.deterministic(self.name, delta + current)
        return apply_limit_fraction(
            proposed, current, limit_val, proposed_val, self.beta
        )


class SurplusFloorHandler:
    def __init__(self, name, prior, beta=20.0):
        self.name = name
        self.prior = prior
        self.beta = beta

    def __call__(self, proposed, state, transform):
        proposed_val = state.eval(transform.expression, with_proposed=proposed)
        delta = numpyro.sample(
            f"delta_{self.name}",
            dist.Normal(self.prior.loc - proposed_val, self.prior.scale),
        )
        threshold = numpyro.deterministic(self.name, delta + proposed_val)
        gate = jax.nn.sigmoid(self.beta * (proposed_val - threshold))
        return {k: gate * v for k, v in proposed.items()}


def assign_transform_handlers(steps, priors, surplus_parameterisation, beta):
    """Assign transform handlers.

    Returns dict of handlers, and dict of param priors, excluding those that are
    sampled by handlers.

    """
    transform_info = {
        transform.name: transform
        for step in steps
        for transform in step.transformations
    }

    if surplus_parameterisation in (None, False):
        surplus_parameterisation = []
    elif surplus_parameterisation is True:
        # Default to all
        surplus_parameterisation = list(transform_info.keys())

    handlers, param_priors = _auto_surplus_transforms(
        surplus_parameterisation, transform_info, priors, beta
    )

    return handlers, param_priors


def default_transform_handler(transform):
    # TODO configure using surplus param by default if possible
    NUMPYRO_DEFAULTS = {
        Floor: NaturalFloorHandler,
        Limit: NaturalLimitHandler,
    }
    return NUMPYRO_DEFAULTS[type(transform)]


def _auto_surplus_transforms(transform_names, transform_info, priors, beta):
    handlers = {}
    params_not_to_sample = set()

    for transform_name in transform_names:
        if transform_name not in transform_info:
            raise ValueError(
                f"Requested reparameterisation of '{transform_name}' but no "
                f"transform with that name exists. "
                f"Available: {list(transform_info.keys())}"
            )

        transform = transform_info[transform_name]

        # Find which symbol in the limit expression is uncertain
        # (has a prior in the spec)
        if isinstance(transform, Limit):
            limit_expr = transform.limit_value
        elif isinstance(transform, Floor):
            limit_expr = transform.threshold_expr
        else:
            raise ValueError(f"Unknown transform type {transform}")

        uncertain_symbols = [s for s in limit_expr.free_symbols if s.name in priors]

        if len(uncertain_symbols) == 0:
            continue
            # In this case the natural parameterisation is fine?
            # raise ValueError(
            #     f"Cannot reparameterise '{transform_name}': its "
            #     f"expression {limit_expr} has no uncertain symbols "
            #     f"(none appear in spec.priors)"
            # )

        if len(uncertain_symbols) > 1:
            raise ValueError(
                f"Automatic surplus reparameterisation for '{transform_name}' "
                f"requires a single uncertain symbol, but found {uncertain_symbols} "
                f"in {limit_expr}. Provide an explicit handler instead."
            )

        sym = uncertain_symbols[0]
        prior = priors[sym.name]

        # Derive the prior on the full limit expression.
        # Substitute recipe data for all other symbols to get
        # limit_expr = a * sym + b (affine in the uncertain symbol).
        # Transform the prior: if sym ~ Normal(mu, sigma),
        # then a*sym + b ~ Normal(a*mu + b, |a|*sigma)
        # TODO: implement this
        # a, b = extract_affine_coefficients(limit_expr, sym, recipe_values)
        # transformed_prior = affine_transform_prior(prior, a, b)
        transformed_prior = prior

        # Create surplus handler
        if isinstance(transform, Limit):
            handlers[transform_name] = SurplusLimitHandler(
                sym.name, transformed_prior, beta
            )
        elif isinstance(transform, Floor):
            handlers[transform_name] = SurplusFloorHandler(
                sym.name, transformed_prior, beta
            )

        # Remove from params_to_sample — the handler owns this symbol
        params_not_to_sample.add(sym.name)

    final_param_priors = {
        str(k): v for k, v in priors.items() if k not in params_not_to_sample
    }
    return handlers, final_param_priors
