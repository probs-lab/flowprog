"""NumPyro compiler helper: convert sympy to JAX expressions
"""

import jax.numpy as jnp
import sympy as sy

from .utils import softplus


def sympy_expr_to_jax(expr, env, structure, constants, beta=100.0):
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
    return _walk(expr, env, structure, constants, beta)


def _walk(expr, env, structure, constants, beta=100.0):

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
        # TODO: better to handle these by callbacks passed in
        base = expr.base
        if base == structure.Balance:
            idx = int(expr.indices[0])
            return _compute_balance_jax(
                structure, structure.objects[idx].id, env, constants
            )
        elif base == structure.ProductionDeficit:
            idx = int(expr.indices[0])
            balance = _compute_balance_jax(
                structure, structure.objects[idx].id, env, constants
            )
            return softplus(-balance, beta)
        elif base == structure.ConsumptionDeficit:
            idx = int(expr.indices[0])
            balance = _compute_balance_jax(
                structure, structure.objects[idx].id, env, constants
            )
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
            result = smooth_max(
                result, _walk(arg, env, structure, constants, beta), beta
            )
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
