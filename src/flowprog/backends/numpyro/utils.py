"""Smooth approximation utilities for the NumPyro compiler.

These replace the discontinuous operations (Max, Piecewise) used by the
sympy compiler with smooth JAX-compatible alternatives.
"""

import jax.numpy as jnp


def smooth_max(a, b, beta=100.0):
    """Smooth approximation to max(a, b) using log-sum-exp.

    As beta -> inf, this approaches the exact max.
    """
    # Use the log-sum-exp trick for numerical stability:
    # max(a, b) ≈ (1/beta) * log(exp(beta*a) + exp(beta*b))
    # = a + (1/beta) * log(1 + exp(beta*(b - a)))  [for stability when a > b]
    m = jnp.maximum(a, b)
    return m + (1.0 / beta) * jnp.log(jnp.exp(beta * (a - m)) + jnp.exp(beta * (b - m)))


def softplus(x, beta=100.0):
    """Smooth approximation to max(0, x).

    softplus(x, beta) = (1/beta) * log(1 + exp(beta * x))

    As beta -> inf, this approaches max(0, x).
    """
    return (1.0 / beta) * jnp.logaddexp(0.0, beta * x)


def smooth_clamp(x, lo, hi, beta=100.0):
    """Smooth approximation to clamp(x, lo, hi).

    Equivalent to smooth_min(smooth_max(x, lo), hi).
    """
    # smooth_max(x, lo) then smooth_min(result, hi)
    # smooth_min(a, b) = -smooth_max(-a, -b)
    above_lo = smooth_max(x, lo, beta)
    return -smooth_max(-above_lo, -hi, beta)
