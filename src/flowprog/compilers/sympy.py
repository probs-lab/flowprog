"""SymPy compiler: compiles model steps into accumulated symbolic expressions.

This is the default compiler used by ModelBuilder._compile().
"""

from collections import defaultdict
import sympy as sy

from ..activities import AdditionalActivity, Limit


def compile_sympy(structure, steps):
    """Compile a list of AdditionalActivity steps into accumulated sympy expressions.

    Walks steps in order, resolving structural symbols (X[j], Y[j],
    Balance[i], ProductionDeficit[i], ConsumptionDeficit[i]) against
    accumulated state and applying transformations.

    :param structure: ModelStructure (provides X, Y, S, U and connectivity)
    :param steps: list[AdditionalActivity]
    :returns: (values dict, intermediates list, history dict)
    """
    values = defaultdict(lambda: sy.S.Zero)
    all_intermediates = []

    for step in steps:
        # 1. Collect intermediates, resolving structural symbols
        for sym, expr, desc in step.intermediates:
            resolved_expr = structure.resolve_structural_symbols(expr, values)
            all_intermediates.append((sym, resolved_expr, desc))

        # 2. Resolve structural symbols in the step's values
        resolved_values = {}
        for sym, expr in step.values.items():
            resolved_values[sym] = structure.resolve_structural_symbols(expr, values)

        # 3. Apply transformations
        for transform in step.transformations:
            if isinstance(transform, Limit):
                resolved_values = _compile_limit(
                    resolved_values, transform, values, structure
                )
            else:
                raise ValueError(f"Unknown transform {transform}")

        # 4. Accumulate
        for sym, expr in resolved_values.items():
            values[sym] += expr

    return dict(values), all_intermediates


def _compile_limit(values_dict, limit_transform, accumulated_values, structure):
    """Compile a Limit transformation into Piecewise expressions.

    The Limit's expression and limit_value contain raw structural symbols.
    The compiler resolves them in two ways:
    - 'current': structural symbols resolved against accumulated state (before this step)
    - 'proposed': structural symbols resolved against accumulated + this step's contribution
    - 'limit': limit_value resolved against accumulated state
    """
    # 'current' = expression evaluated with accumulated values only
    current = structure.resolve_structural_symbols(
        limit_transform.expression, accumulated_values
    )

    # 'limit' = limit_value evaluated with accumulated values
    limit_resolved = structure.resolve_structural_symbols(
        limit_transform.limit_value, accumulated_values
    )

    # 'proposed' = expression evaluated with (accumulated + step contribution)
    M = len(structure.processes)
    proposed_values = dict(accumulated_values)
    for j in range(M):
        xj = structure.X[j]
        yj = structure.Y[j]
        if xj in values_dict:
            proposed_values[xj] = (
                accumulated_values.get(xj, sy.S.Zero) + values_dict[xj]
            )
        if yj in values_dict:
            proposed_values[yj] = (
                accumulated_values.get(yj, sy.S.Zero) + values_dict[yj]
            )

    proposed = structure.resolve_structural_symbols(
        limit_transform.expression, proposed_values
    )

    diff = proposed - current

    # Epsilon protection for division by zero (same as original)
    epsilon = sy.S(10) ** -10
    safe_diff = sy.Max(diff, epsilon)

    return {
        k: sy.Piecewise(
            (sy.S.Zero, current >= limit_resolved),
            (v, proposed <= limit_resolved),
            ((limit_resolved - current) / safe_diff * v, True),
            evaluate=False,
        )
        for k, v in values_dict.items()
    }
