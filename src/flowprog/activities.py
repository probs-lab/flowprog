"""Data types for representing model-building steps.

An AdditionalActivity describes a proposed change to process activities,
produced by helper functions like pull_production() and push_consumption().
It carries the sympy expressions alongside metadata (intermediates,
transformations) that enable alternative compilation backends.
"""

from dataclasses import dataclass, field
from collections import defaultdict
from collections.abc import Iterator
from typing import Optional, Union
import sympy as sy


# ── Activity transformations ──────────────────────────────────────────


@dataclass(frozen=True)
class Limit:
    """Constrain an expression not to exceed a bound.

    Both expression and limit_value contain raw structural symbols
    (X[j], Y[j], Balance[i], ProductionDeficit[i], etc.) that the
    compiler resolves against accumulated state.
    """

    expression: sy.Expr
    limit_value: sy.Expr
    name: str | None = None


@dataclass(frozen=True)
class Floor:
    """Constrain an expression to be at least a threshold, or else zero.

    If the *proposed* value of `expression` (accumulated + this step's
    contribution) would be below `threshold`, the entire step is scaled
    to zero.  Otherwise the step is kept as-is.

    This implements a "minimum turndown" constraint: either the process
    operates above a minimum level, or it doesn't operate at all.

    Both expression and threshold contain raw structural symbols that the
    compiler resolves against accumulated state.
    """

    expression: sy.Expr
    threshold: sy.Expr
    name: str | None = None


# Union type for all transformations (extensible with Clamp, etc.)
ActivityTransformation = Union[Limit, Floor]


@dataclass
class AdditionalActivity:
    """A proposed change to process activities, with optional transformations.

    Produced by pull_production(), push_consumption(), etc.
    Committed to the model via add().

    :param values: Mapping from model symbols (X[j], Y[j]) to their
        contributed expressions.
    :param intermediates: Intermediate symbol definitions for CSE,
        as (symbol, expression, description) triples.
    :param transformations: Ordered list of transformations to apply
        before committing (e.g. Limit).
    :param description: Human-readable label for this step.
    """

    values: dict[sy.Expr, sy.Expr] = field(default_factory=dict)
    intermediates: list[tuple[sy.Symbol, sy.Expr, str]] = field(default_factory=list)
    transformations: list[ActivityTransformation] = field(default_factory=list)
    description: Optional[str] = None

    def __getitem__(self, key):
        """Allow dict-style access to values for backward compatibility."""
        return self.values[key]


def _serialize_transformation(transform):
    """Serialize an ActivityTransformation to a JSON-friendly dict."""
    if isinstance(transform, Limit):
        return {
            "type": "Limit",
            "expression": sy.srepr(transform.expression),
            "limit_value": sy.srepr(transform.limit_value),
            "name": transform.name,
        }
    if isinstance(transform, Floor):
        return {
            "type": "Floor",
            "expression": sy.srepr(transform.expression),
            "threshold": sy.srepr(transform.threshold),
            "name": transform.name,
        }
    raise TypeError(f"Unknown transformation type: {type(transform)}")


def _deserialize_transformation(data, namespace):
    """Deserialize a transformation dict back to an ActivityTransformation."""
    if data["type"] == "Limit":
        return Limit(
            expression=sy.sympify(data["expression"], locals=namespace),
            limit_value=sy.sympify(data["limit_value"], locals=namespace),
            name=data["name"],
        )
    if data["type"] == "Floor":
        return Floor(
            expression=sy.sympify(data["expression"], locals=namespace),
            threshold=sy.sympify(data["threshold"], locals=namespace),
            name=data["name"],
        )
    raise ValueError(f"Unknown transformation type: {data['type']}")


def serialize_step(step: AdditionalActivity) -> dict:
    """Serialize an AdditionalActivity to a JSON-friendly dict."""
    return {
        "values": {sy.srepr(k): sy.srepr(v) for k, v in step.values.items()},
        "intermediates": [
            {
                "symbol": sy.srepr(sym),
                "expr": sy.srepr(expr),
                "label": label,
            }
            for sym, expr, label in step.intermediates
        ],
        "transformations": [_serialize_transformation(t) for t in step.transformations],
        "description": step.description,
    }


def deserialize_step(data: dict, namespace: dict) -> AdditionalActivity:
    """Deserialize a dict back to an AdditionalActivity."""
    values = {
        sy.sympify(k, locals=namespace): sy.sympify(v, locals=namespace)
        for k, v in data["values"].items()
    }
    intermediates = [
        (
            sy.sympify(item["symbol"], locals=namespace),
            sy.sympify(item["expr"], locals=namespace),
            item["label"],
        )
        for item in data["intermediates"]
    ]
    transformations = [
        _deserialize_transformation(t, namespace) for t in data["transformations"]
    ]
    return AdditionalActivity(
        values=values,
        intermediates=intermediates,
        transformations=transformations,
        description=data.get("description"),
    )


## Helpers


def create_intermediate(
    intermediates: list[tuple[sy.Symbol, sy.Expr, str]],
    value: sy.Expr,
    description: str,
    counter: Iterator[sy.Symbol],
) -> sy.Symbol:
    """Create an intermediate symbol for a value (pure — appends to provided list).

    Returns the new symbol.
    """
    new_sym = next(counter)
    intermediates.append((new_sym, value, description))
    return new_sym


def eval_activity_expr(
    expr: sy.Expr, activity: "AdditionalActivity", values=None
) -> sy.Expr:
    """Evaluate an expression using an activity's intermediates.

    Substitutes intermediate definitions and optional extra values.
    """
    if values is None:
        values = {}
    intermediates = [
        (
            sym,
            sym_value.xreplace(values) if isinstance(sym_value, sy.Expr) else sym_value,
        )
        for sym, sym_value, _ in activity.intermediates
    ]
    return expr.subs(intermediates[::-1])


def merge_activities(*activities: AdditionalActivity) -> AdditionalActivity:
    """Merge multiple AdditionalActivities by summing their values."""
    merged_values: dict[sy.Expr, sy.Expr] = defaultdict(lambda: sy.S.Zero)
    merged_intermediates: list[tuple[sy.Symbol, sy.Expr, str]] = []
    merged_transformations: list[ActivityTransformation] = []

    for act in activities:
        for k, v in act.values.items():
            merged_values[k] += v
        merged_intermediates.extend(act.intermediates)
        merged_transformations.extend(act.transformations)

    return AdditionalActivity(
        values=dict(merged_values),
        intermediates=merged_intermediates,
        transformations=merged_transformations,
    )
