"""Markdown compiler: produces a human-readable description of model steps.

Useful for debugging and understanding the model-building process.
Also serves as a minimal reference implementation for new compilers.
"""

import sympy as sy

from ..activities import Limit


def compile_markdown(structure, steps):
    """Compile model steps into a Markdown-formatted description.

    :param structure: ModelStructure (provides process/object metadata)
    :param steps: list[AdditionalActivity]
    :returns: Markdown string describing the model and its steps
    """
    lines = []
    lines.append("# Model description")
    lines.append("")
    lines.append(_format_structure(structure))

    if not steps:
        lines.append("*No steps recorded.*")
        return "\n".join(lines)

    lines.append("## Steps")
    lines.append("")

    for i, step in enumerate(steps, 1):
        lines.append(_format_step(i, step, structure))

    return "\n".join(lines)


def _format_structure(structure):
    """Format the model structure section."""
    lines = []
    lines.append("## Structure")
    lines.append("")

    lines.append(f"**Processes** ({len(structure.processes)}):")
    lines.append("")
    for j, p in enumerate(structure.processes):
        consumes = ", ".join(p.consumes) if p.consumes else "(none)"
        produces = ", ".join(p.produces) if p.produces else "(none)"
        stock = " (has stock)" if p.has_stock else ""
        lines.append(
            f"- **{p.id}** (index {j}): consumes [{consumes}] → produces [{produces}]{stock}"
        )
    lines.append("")

    lines.append(f"**Objects** ({len(structure.objects)}):")
    lines.append("")
    for i, o in enumerate(structure.objects):
        market = " (has market)" if o.has_market else ""
        lines.append(f"- **{o.id}** (index {i}){market}")
    lines.append("")

    return "\n".join(lines)


def _format_step(step_num, step, structure):
    """Format a single step."""
    lines = []

    label = step.description or "(unlabelled)"
    lines.append(f"### Step {step_num}: {label}")
    lines.append("")

    # Describe affected symbols
    if step.values:
        lines.append("**Assignments:**")
        lines.append("")
        for sym, expr in step.values.items():
            desc = _describe_symbol(sym, structure)
            lines.append(f"- {desc} += `{expr}`")
        lines.append("")

    # Intermediates
    if step.intermediates:
        lines.append("**Intermediates:**")
        lines.append("")
        for sym, expr, desc in step.intermediates:
            lines.append(f"- `{sym}` = `{expr}` — {desc}")
        lines.append("")

    # Transformations
    if step.transformations:
        lines.append("**Transformations:**")
        lines.append("")
        for t in step.transformations:
            lines.append(f"- {_describe_transformation(t, structure)}")
        lines.append("")

    # Free parameters
    params = _collect_free_symbols(step, structure)
    if params:
        names = ", ".join(f"`{s}`" for s in sorted(params, key=str))
        lines.append(f"**Parameters:** {names}")
        lines.append("")

    return "\n".join(lines)


def _describe_symbol(sym, structure):
    """Produce a human-readable description for a model symbol like X[0] or Y[1]."""
    if not isinstance(sym, sy.Indexed):
        return f"`{sym}`"

    idx = int(sym.indices[0])
    if sym.base == structure.X:
        proc = structure.processes[idx]
        return f"`X[{idx}]` (input activity of **{proc.id}**)"
    elif sym.base == structure.Y:
        proc = structure.processes[idx]
        return f"`Y[{idx}]` (output activity of **{proc.id}**)"
    return f"`{sym}`"


def _describe_transformation(t, structure):
    """Produce a human-readable description of a transformation."""
    if isinstance(t, Limit):
        return f"Limit: `{t.expression}` must not exceed `{t.limit_value}`"
    return f"Unknown transformation: {t!r}"


def _collect_free_symbols(step, structure):
    """Collect user-defined free symbols (excluding model indexed bases)."""
    model_bases = {
        structure.X,
        structure.Y,
        structure.S,
        structure.U,
        structure.Balance,
        structure.ProductionDeficit,
        structure.ConsumptionDeficit,
    }
    # Symbol names corresponding to model IndexedBase objects
    model_base_names = {b.label.name for b in model_bases}

    params = set()
    all_exprs = list(step.values.values())
    all_exprs.extend(expr for _, expr, _ in step.intermediates)
    for t in step.transformations:
        if isinstance(t, Limit):
            all_exprs.extend([t.expression, t.limit_value])

    for expr in all_exprs:
        if not isinstance(expr, sy.Basic):
            continue
        # Plain symbols (not Indexed)
        params.update(expr.free_symbols)

    # Collect intermediate symbol names to exclude
    intermediate_syms = {sym for sym, _, _ in step.intermediates}

    # Filter: keep only plain Symbols whose name isn't a model IndexedBase
    # and that aren't intermediate symbols (x0, x1, ...)
    params = {
        s
        for s in params
        if isinstance(s, sy.Symbol)
        and s.name not in model_base_names
        and s not in intermediate_syms
    }
    return params
