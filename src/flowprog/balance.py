"""Since flowprog allows models to be built up in steps, before all steps are
complete then it is natural that not all parts of the model will balance (e.g.
some objects may have an excess of production compared to their consumption). To
check that the final model's markets do indeed balance where they should,
flowprog tracks this through each step and reports the results for the compiled
model via a :py:class:`~flowprog.BalanceTrace`:

.. code-block:: python

   model = builder.build(recipe_data)

   model.balance_trace.markets()                    # every market
   model.balance_trace.markets(Verdict.OPEN)        # the ones it cannot balance
   print(model.balance_trace.explain("Electricity"))  # why, step by step

A market is reported as :py:attr:`flowprog.balance.Verdict.BALANCED` if it is
known to balance for *every* parameter value. :py:attr:`CONDIIONAL
<flowprog.balance.Verdict.CONDITIONAL>` means it balances for some parameter
values and not others, and :py:attr:`OPEN <flowprog.balance.Verdict.OPEN>` means
that, as far as can be determined, the market does not balance at all.

How a market is proved to balance
---------------------------------

Compiling walks the model's steps in order, accumulating what each contributes
to the process activities ``X[j]`` and ``Y[j]``. Alongside that it accumulates
each object's balance,

.. math:: \\sum_j S_{ij} Y_j - \\sum_j U_{ij} X_j

and a :class:`Sign` for it. The sign is what makes a proof possible, since
object production/consumption deficits are normally discontinuous at zero, but
if the sign of the balance is known then this can be simplified.

If a market does not balance, its *residual* is reported by
:meth:`BalanceTrace.explain <flowprog.BalanceTrace.explain>`.

"""

import logging
from dataclasses import dataclass
from itertools import combinations
from enum import Enum
from typing import Optional

import pandas as pd
import sympy as sy

_log = logging.getLogger(__name__)


class Verdict(Enum):
    """How well a market balances."""

    BALANCED = "balanced"
    """Production equals consumption for *every* parameter value."""

    CONDITIONAL = "conditional"
    """The market balances for some parameter values and not others. This may be
    because a capacity limit may or may not bind, or because it balances only
    when certain parameters satisfy an identity, such as needing shares to sum
    to one. Also reported when the analysis could not reach a conclusion either
    way.

    """

    OPEN = "open"
    """Production never equals consumption."""

    def __str__(self):
        return self.value


class Effect(Enum):
    """The effect of one step on one object's balance."""

    OPENS = "opens"
    """The step left the object out of balance."""

    CLOSES = "closes"
    """The step brought the object back into balance exactly."""

    CLOSES_UNLESS_CONSTRAINED = "closes unless constrained"
    """The step would have closed the object, but a transformation (e.g. from
    :py:meth:`~flowprog.ModelBuilder.limit`) may constrain the effect of the
    step and prevent the balance reaching zero, depending on parameter values.

    """

    def __str__(self):
        return self.value


class Sign(Enum):
    """The sign of an expression, as far as it can be determined.
    """

    ZERO = "zero"
    """Exactly zero."""

    NON_NEGATIVE = "non-negative"
    """A non-negative balance (supply may exceed demand)."""

    NON_POSITIVE = "non-positive"
    """A non-positive balance (demand may exceed supply)."""

    UNKNOWN = "unknown"
    """The sign could not be determined."""

    def __str__(self):
        return self.value

    @property
    def opposite(self):
        if self is Sign.NON_NEGATIVE:
            return Sign.NON_POSITIVE
        if self is Sign.NON_POSITIVE:
            return Sign.NON_NEGATIVE
        return self


# Depth limit when following chains of intermediate symbol definitions, so an
# unexpected cycle cannot hang the caller.
_MAX_DEPTH = 100

# How large an expression may get, in tree nodes, before it is not worth
# analysing further. Writing out the values a balance is built from is what
# makes cancellation visible, but for an object that never comes back into
# balance each further step can embed another copy of what is outstanding, and
# the expression grows without bound. Past this size the object is reported as
# conditional, with a reason saying the analysis stopped early: an expression
# that large has not cancelled and is not going to, but neither is it a
# faithful account of what is missing.
MAX_NODES = 500


def too_large(expr, limit=MAX_NODES):
    """Whether an expression has grown past the point of being worth analysing.

    Counts nodes with an early exit, rather than using ``sympy.count_ops``,
    which is thorough but far too slow to call on every step of a build.
    """
    if not isinstance(expr, sy.Basic):
        return False
    stack, count = [expr], 0
    while stack:
        count += 1
        if count > limit:
            return True
        stack.extend(stack.pop().args)
    return False


def sign_of(expr, definitions=None, cache=None, _depth=0):
    """Determine the sign of `expr`, as far as that is possible.

    :param expr: Sympy expression.
    :param definitions: Optional mapping from intermediate symbols to the
        expressions they stand for; symbols found here are followed.
    :param cache: Optional dict, reused across calls that share the same
        `definitions`, so that subexpressions are only worked out once.
    :return: A :class:`Sign`.

    Deliberately conservative: :attr:`Sign.UNKNOWN` means "cannot tell", not
    "varies in sign". Process activities and recipe coefficients are declared
    non-negative by :class:`~flowprog.ModelStructure`, so most flow expressions
    resolve without anything having to be declared about model parameters.

    Sympy's own assumptions are consulted, but only as a last resort, and they
    are not enough on their own: they cannot see through the intermediate
    symbols a compiled model is largely written in, and they decline to give a
    sign to anything divided by a coefficient that might be zero.
    """
    if definitions is None:
        definitions = {}
    if _depth > _MAX_DEPTH:
        return Sign.UNKNOWN
    if not isinstance(expr, sy.Basic):
        expr = sy.S(expr)
    if cache is not None and expr in cache:
        return cache[expr]

    sign = _structural_sign(expr, definitions, cache, _depth)
    if sign is Sign.UNKNOWN:
        # Fall back to sympy's own assumptions, which is where a declaration
        # like `nonnegative=True` on a symbol is picked up. Left until last
        # because deducing it is far more expensive than looking at the shape
        # of the expression, and most expressions here are settled by shape.
        if expr.is_nonnegative:
            sign = Sign.NON_NEGATIVE
        elif expr.is_nonpositive:
            sign = Sign.NON_POSITIVE

    if cache is not None:
        cache[expr] = sign
    return sign


def _structural_sign(expr, definitions, cache, depth):
    """The sign of `expr` as far as its shape alone determines it."""
    if expr.is_number:
        if expr.is_zero:
            return Sign.ZERO
        return Sign.NON_NEGATIVE if expr > 0 else Sign.NON_POSITIVE

    def sign(sub):
        return sign_of(sub, definitions, cache, depth + 1)

    if isinstance(expr, sy.Symbol):
        return sign(definitions[expr]) if expr in definitions else Sign.UNKNOWN

    if isinstance(expr, sy.Add):
        return _common_sign({sign(a) for a in expr.args})

    if isinstance(expr, sy.Mul):
        result = Sign.NON_NEGATIVE
        for arg in expr.args:
            arg_sign = sign(arg)
            if arg_sign is Sign.ZERO:
                return Sign.ZERO
            if arg_sign is Sign.UNKNOWN:
                return Sign.UNKNOWN
            if arg_sign is Sign.NON_POSITIVE:
                result = result.opposite
        return result

    if isinstance(expr, sy.Pow):
        base = sign(expr.base)
        if expr.exp.is_number and (expr.exp.is_negative or expr.exp.is_odd):
            # 1/x and odd powers keep the sign of x. For a negative exponent
            # that takes x to be non-zero, which sympy will not assume -- but a
            # recipe coefficient of zero would mean the model had divided by
            # zero long before this.
            return base
        if base in (Sign.NON_NEGATIVE, Sign.ZERO):
            return Sign.NON_NEGATIVE
        return Sign.UNKNOWN

    if isinstance(expr, (sy.Max, sy.Min)):
        signs = [sign(a) for a in expr.args]
        wins = Sign.NON_NEGATIVE if isinstance(expr, sy.Max) else Sign.NON_POSITIVE
        if wins in signs or Sign.ZERO in signs:
            return wins
        if all(s is wins.opposite for s in signs):
            return wins.opposite
        return Sign.UNKNOWN

    if isinstance(expr, sy.Piecewise):
        return _common_sign({sign(value) for value, _ in expr.args})

    return Sign.UNKNOWN


def _common_sign(signs):
    """The sign shared by a set of terms, ignoring those that are zero."""
    signs = signs - {Sign.ZERO}
    if not signs:
        return Sign.ZERO
    if signs <= {Sign.NON_NEGATIVE}:
        return Sign.NON_NEGATIVE
    if signs <= {Sign.NON_POSITIVE}:
        return Sign.NON_POSITIVE
    return Sign.UNKNOWN


def combine_signs(a, b):
    """The sign of a sum, given the signs of its two terms."""
    if a is Sign.ZERO:
        return b
    if b is Sign.ZERO:
        return a
    return a if a is b else Sign.UNKNOWN


@dataclass(frozen=True)
class BalanceEvent:
    """One step's effect on one object's balance.

    :param step: Index of the step in the model.
    :param effect: What the step did -- see :class:`Effect`.
    :param description: The step's label, if it was given one.
    :param sign_before: Sign of the balance before this step.
    :param sign_after: Sign of the balance after this step.
    """

    step: int
    effect: Effect
    description: Optional[str]
    sign_before: Sign
    sign_after: Sign


class BalanceTrace:
    """Accumulated balances for each object's market, and events that contributed.

    This can determine which markets can be shown to balance by construction
    (see :meth:`verdict`), and for those that don't, which model steps were
    involved (see :meth:`explain`).

    """

    def __init__(
        self,
        structure,
        balances,
        signs=None,
        events=None,
        incomplete=(),
        definitions=None,
        numerical_guards=(),
    ):
        """
        :param structure: The model structure these balances belong to.
        :param balances: ``{object index: production - consumption}`` left over.
        :param signs: ``{object index: Sign}`` of each balance, where it is
            known. Missing entries count as :attr:`Sign.UNKNOWN`.
        :param events: ``{object index: (BalanceEvent, ...)}`` in step order.
        :param incomplete: Object indices whose balance grew too large to
            follow while compiling, so it accounts for only part of what is
            outstanding there.
        :param definitions: What each intermediate symbol appearing in the
            balances stands for, needed to follow them in :meth:`breakpoints`.
        :param numerical_guards: Values a compiler introduces for numerical
            safety, such as a small number keeping a division finite. They look
            like bounds but do not mark a change in what the model does, so
            :meth:`breakpoints` leaves them out.
        """
        self.structure = structure
        self.balances = dict(balances)
        self.signs = dict(signs or {})
        self.events = {i: tuple(e) for i, e in (events or {}).items()}
        self.incomplete = set(incomplete)
        self._definitions = definitions or {}
        self._numerical_guards = tuple(numerical_guards)
        self._index = {obj.id: i for i, obj in enumerate(structure.objects)}
        self._verdicts: dict[str, tuple[Verdict, str]] = {}

    def __repr__(self):
        counted = ", ".join(
            f"{len(self.markets(verdict))} {verdict}" for verdict in Verdict
        )
        return f"BalanceTrace({counted})"

    def markets(self, with_verdict=None):
        """Ids of the objects that are supposed to balance, in model order.

        :param with_verdict: Optional :class:`Verdict` to restrict to, so that
            ``markets(Verdict.OPEN)`` is the list of markets the model cannot
            balance at all.
        """
        ids = [obj.id for obj in self.structure.objects if obj.has_market]
        if with_verdict is None:
            return ids
        return [o for o in ids if self.verdict(o)[0] is with_verdict]

    def residual(self, object_id):
        """Net production minus consumption for `object_id`."""
        return self.balances.get(self._index[object_id], sy.S.Zero)

    def sign(self, object_id):
        """The :class:`Sign` of `object_id`'s balance, as far as it is known."""
        return self.signs.get(self._index[object_id], Sign.UNKNOWN)

    def events_for(self, object_id):
        """The :class:`BalanceEvent` objects affecting `object_id`, in order."""
        return self.events.get(self._index[object_id], ())

    def verdict(self, object_id):
        """Classify one market: ``(verdict, reason)``.

        The verdict is a :class:`Verdict`, and the reason is a human-readable
        explanation.

        """
        if object_id not in self._verdicts:
            self._verdicts[object_id] = self._classify(object_id)
        return self._verdicts[object_id]

    def to_dataframe(self):
        """One row per market: object, verdict, reason, and residual expression."""
        rows = [
            (object_id, *self.verdict(object_id), self.residual(object_id))
            for object_id in self.markets()
        ]
        return pd.DataFrame(rows, columns=["object", "verdict", "reason", "residual"])

    def explain(self, object_id):
        """A step-by-step account of how an object's market ended up.

        Lists every step that changed this object's balance, what it did, and
        the residual left over.
        """
        verdict, reason = self.verdict(object_id)
        lines = [f"{object_id}: {verdict} -- {reason}"]
        for event in self.events_for(object_id):
            label = event.description or "(unlabelled step)"
            lines.append(f"  step {event.step:3d}  {str(event.effect):24s} {label}")
        residual = self.residual(object_id)
        if residual != 0:
            lines.append(f"  residual: {residual}")
        return "\n".join(lines)

    def breakpoints(self, object_id):
        """Conditions under which `object_id` switches between balancing or not.

        Each is an inequality -- a capacity that may or may not bind, a minimum
        operating threshold, or a deficit that may or may not be positive. Which
        side of them the model is on decides whether this market balances. Empty
        for a market that balances unconditionally, and for one that never
        balances.

        Values passed as `numerical_guards` are skipped as breakpoints.

        Enumerating breakpoints for an expression is relatively slow, so they
        are calculated here on demand rather than during compilation.

        """
        conditions, seen = [], set()
        for condition in self._walk_conditions(object_id):
            key = sy.srepr(condition)
            if key not in seen:
                seen.add(key)
                conditions.append(condition)
        return conditions

    # -- serialisation --

    def to_dict(self):
        """A JSON-friendly summary, one entry per object that has a balance."""
        objects = self.structure.objects
        out = {}
        for i, balance in self.balances.items():
            entry = {"balance": sy.srepr(sy.S(balance))}
            sign = self.signs.get(i)
            if sign is not None:
                entry["sign"] = sign.value
            if i in self.incomplete:
                entry["incomplete"] = True
            events = self.events.get(i)
            if events:
                entry["events"] = [
                    {
                        "step": event.step,
                        "effect": event.effect.value,
                        "description": event.description,
                        "sign_before": event.sign_before.value,
                        "sign_after": event.sign_after.value,
                    }
                    for event in events
                ]
            out[objects[i].id] = entry
        return out

    @classmethod
    def from_dict(cls, data, structure, definitions=None, numerical_guards=(),
                  sympify=sy.sympify):
        """Rebuild a trace from :meth:`to_dict` output.

        :param sympify: How to turn a saved expression back into sympy, if the
            symbols involved need a namespace to be recognised.
        """
        index = {obj.id: i for i, obj in enumerate(structure.objects)}
        balances, signs, events, incomplete = {}, {}, {}, set()
        for object_id, entry in data.items():
            if object_id not in index:
                continue
            i = index[object_id]
            balances[i] = sympify(entry["balance"])
            if "sign" in entry:
                signs[i] = Sign(entry["sign"])
            if entry.get("incomplete"):
                incomplete.add(i)
            if entry.get("events"):
                events[i] = tuple(
                    BalanceEvent(
                        step=event["step"],
                        effect=Effect(event["effect"]),
                        description=event["description"],
                        sign_before=Sign(event["sign_before"]),
                        sign_after=Sign(event["sign_after"]),
                    )
                    for event in entry["events"]
                )
        return cls(
            structure,
            balances,
            signs,
            events,
            incomplete,
            definitions,
            numerical_guards,
        )

    # -- internals --

    def _classify(self, object_id):
        residual = self.residual(object_id)
        if residual == 0:
            return Verdict.BALANCED, "balances for all parameter values"

        incomplete = self._index[object_id] in self.incomplete
        if self._has_condition(object_id):
            sign = self.sign(object_id)
            if sign is Sign.NON_POSITIVE:
                detail = "demand can go unmet"
            elif sign is Sign.NON_NEGATIVE:
                detail = "supply can go unabsorbed"
            else:
                detail = "can be out of balance in either direction"
            reason = (
                f"balances in some parameter regimes only -- {detail}; "
                "see breakpoints()"
            )
            if incomplete:
                reason += " (residual shows only part of what is outstanding)"
            return Verdict.CONDITIONAL, reason

        if incomplete:
            return Verdict.CONDITIONAL, (
                "not proved to balance: what is outstanding grew too large to "
                "follow, so the residual shows only part of it"
            )

        identities = self._identities(object_id)
        if identities:
            joined = " = 0, or ".join(str(identity) for identity in identities)
            return Verdict.CONDITIONAL, f"balances only where {joined} = 0"

        return Verdict.OPEN, (
            "never balances: production and consumption cannot be equal"
        )

    def _walk_conditions(self, object_id, first_only=False):
        """Yield the branch conditions the residual for `object_id` depends on.

        Follows intermediate symbols as a directed graph, visiting each
        definition once. Substituting them instead is not viable: the expanded
        expressions can grow by orders of magnitude.
        """
        visited = set()
        stack = [self.residual(object_id)]
        while stack:
            expr = stack.pop()
            if not isinstance(expr, sy.Basic):
                continue
            if isinstance(expr, sy.Symbol):
                if expr in self._definitions and expr not in visited:
                    visited.add(expr)
                    stack.append(self._definitions[expr])
                continue
            if isinstance(expr, sy.Piecewise):
                for value, condition in expr.args:
                    stack.append(value)
                    if condition is not sy.true:
                        yield condition
                        if first_only:
                            return
                        stack.append(condition)
                continue
            if isinstance(expr, (sy.Max, sy.Min)):
                # Whichever argument wins decides what the model does, so each
                # pair of them is a boundary -- unless one is a guard the
                # compiler added for numerical safety, which looks like a bound
                # but is not one.
                if not any(guard in expr.args for guard in self._numerical_guards):
                    for a, b in combinations(expr.args, 2):
                        yield sy.Ge(a, b, evaluate=False)
                        if first_only:
                            return
            stack.extend(expr.args)

    def _has_condition(self, object_id):
        """Whether the residual depends on any branch at all (cheap check)."""
        for _ in self._walk_conditions(object_id, first_only=True):
            return True
        return False

    def _identities(self, object_id):
        """Parameter identities that would make a branch-free residual vanish.

        A model that splits a flow between routes using share parameters
        balances only if those shares sum to one -- a claim about parameters
        rather than about model structure, so the residual factorises through
        something like ``share_a + share_b - 1``.
        """
        residual = self.residual(object_id)
        if too_large(residual):
            return []

        identities = []
        for factor in sy.factor(sy.cancel(residual)).as_powers_dict():
            if not isinstance(factor, sy.Add):
                continue
            if sign_of(factor, self._definitions) is not Sign.UNKNOWN:
                continue
            # Keep affine forms like `share_a + share_b - 1`: every term is a
            # number, or a number times a single symbol.
            terms = sy.Add.make_args(factor)
            if all(t.is_number or t.as_coeff_Mul()[1].is_Atom for t in terms):
                identities.append(factor)
        return identities


def log_summary(trace, logger: Optional[logging.Logger] = None):
    """Log a one-line summary of a trace.

    Called after compiling a model, so a market the model does not balance is
    noticed without anyone having to ask.
    """
    logger = logger or _log
    markets = trace.markets()
    unproved = [o for o in markets if trace.residual(o) != 0]
    if unproved:
        shown = ", ".join(sorted(unproved)[:5])
        if len(unproved) > 5:
            shown += ", ..."
        logger.warning(
            "%d of %d markets are not proved to balance (%s). Inspect with "
            "model.balance_trace.explain(object_id).",
            len(unproved),
            len(markets),
            shown,
        )
