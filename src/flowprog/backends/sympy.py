"""SymPy compiler: compiles model steps into accumulated symbolic expressions.

This is the default compiler used by ModelBuilder._compile().
"""

import logging
from typing import Container, Optional, Union
from collections import defaultdict
import sympy as sy
import numpy as np
import pandas as pd
from rdflib import URIRef

from ..model_structure import (
    ModelStructure,
    Process,
    Object,
    ElementaryExchange,
    flow_id,
)
from ..activities import AdditionalActivity, Limit, Floor, create_intermediate
from ..balance import (
    BalanceEvent,
    BalanceTrace,
    Effect,
    Sign,
    combine_signs,
    log_summary,
    sign_of,
    too_large,
)

_log = logging.getLogger(__name__)

# Prefix for intermediate symbols minted by the compiler itself, as opposed to
# those carried on the steps by ModelBuilder (which uses the default "x"
# prefix of sy.numbered_symbols()).
_COMPILER_INTERMEDIATE_PREFIX = "_t"

# Expression types that are already O(1) and cannot grow with accumulated
# state, so are cheaper left inline than routed through an intermediate.
_TRIVIAL_TYPES = (sy.Symbol, sy.Number, sy.Indexed)


# Guard against dividing by zero when a capacity limit works out how far to
# scale a step down. Small enough not to affect the answer where the division
# is meaningful, and only reached where it is not.
_LIMIT_EPSILON = sy.S(10) ** -10


def _object_balance(structure, i, contributions):
    """Production minus consumption of object `i` from a set of activities."""
    produced = sum(
        (
            structure.S[i, j] * contributions.get(structure.Y[j], sy.S.Zero)
            for j in structure.processes_producing(i)
        ),
        sy.S.Zero,
    )
    consumed = sum(
        (
            structure.U[i, j] * contributions.get(structure.X[j], sy.S.Zero)
            for j in structure.processes_consuming(i)
        ),
        sy.S.Zero,
    )
    return produced - consumed


def _resolve_placeholders(structure, expr, values, balances, signs, proposed=None):
    """Replace structural placeholder symbols with their current value.

    This helper function is shared by :class:`SympyCompiler`, which resolves
    placeholders step by step as it goes, and :class:`SympyModel`, which
    resolves them afterwards against the final accumulated state.

    :param values: ``{X[j] or Y[j]: expression}`` accumulated so far.
    :param balances: ``{i: expression}`` balances accumulated so far.
    :param signs: ``{i: Sign}`` of those balances. A known sign allows
        balance expressions to be potentially simplified.
    :param proposed: Optional ``{X[j] or Y[j]: contribution}`` added to
        ``values`` before resolving placeholders.

    """
    if not isinstance(expr, sy.Basic):
        return expr

    def activity(symbol):
        value = values.get(symbol, sy.S.Zero)
        if proposed is not None and symbol in proposed:
            return value + proposed[symbol]
        return value

    def balance(i):
        value = balances.get(i, sy.S.Zero)
        if proposed is not None:
            value = value + _object_balance(structure, i, proposed)
        return value

    def balance_sign(i):
        # A proposal is a hypothetical; no sign has been worked out for it.
        # TODO: can this be usefully tightened?
        if proposed is not None:
            return Sign.UNKNOWN
        return signs.get(i, Sign.UNKNOWN)

    def shortfall(amount, sign):
        """``Max(0, amount)``, without the ``Max`` where the sign is known."""
        if sign in (Sign.NON_NEGATIVE, Sign.ZERO):
            return amount
        if sign is Sign.NON_POSITIVE:
            return sy.S.Zero
        return sy.Max(0, amount, evaluate=False)

    subs = {}
    for sym in expr.atoms(sy.Indexed):
        # Compared by equality, not identity: sympy caches Indexed objects
        # globally, so `Y[0]` can come back carrying an equal but distinct
        # IndexedBase created by another ModelStructure.
        base = sym.base
        if base == structure.X or base == structure.Y:
            subs[sym] = activity(sym)
        elif base == structure.Balance:
            subs[sym] = balance(sym.indices[0])
        elif base == structure.ProductionDeficit:
            # Production is short by however far the balance is below zero.
            i = sym.indices[0]
            subs[sym] = shortfall(-balance(i), balance_sign(i).opposite)
        elif base == structure.ConsumptionDeficit:
            # Consumption is short by however far the balance is above zero.
            i = sym.indices[0]
            subs[sym] = shortfall(balance(i), balance_sign(i))
        elif base == structure.ElementaryBalance:
            e = sym.indices[0]
            subs[sym] = sum(
                (
                    structure.B[e, j] * activity(structure.Y[j])
                    for j in structure.processes_with_exchange(e)
                ),
                sy.S.Zero,
            )
    return expr.xreplace(subs) if subs else expr


class SympyCompiler:
    """Turns a model's steps into the accumulated expressions that define it.

    The main method is :meth:`compile`, which walks the steps in order, resolves
    each step's placeholder symbols against what has been accumulated so far,
    applies transformations, and adds the transformed contributions to the
    state, then returns a :class:`SympyModel`.

    Object balances are accumulated alongside the process activities, together
    with their signs. This allows expressions to be simplified in some cases,
    and tracing of whether object balances close.

    This is a separate object from the model-builder and the final model because
    it has additional caches and bookkeeping only needed during compilation.

    """

    def __init__(self, structure: ModelStructure):
        self.structure = structure
        self.values = defaultdict(lambda: sy.S.Zero)
        self.intermediates: list[tuple[sy.Symbol, sy.Expr, str]] = []

        # Balances are kept in two forms: the compact one keeps intermediate
        # expressions unexpanded for use in later model steps, while the
        # written-out one expands them to make cancellations visible.
        #
        # Every object starts at a known balance of zero.
        objects = range(len(structure.objects))
        self.balances = {i: sy.S.Zero for i in objects}
        self.written_out = {i: sy.S.Zero for i in objects}
        self.signs = {i: Sign.ZERO for i in objects}
        self.events = defaultdict(list)
        self.incomplete: set[int] = set()

        self._definitions: dict[sy.Symbol, sy.Expr] = {}
        self._expansions: dict[sy.Symbol, sy.Expr] = {}
        self._sign_cache: dict[sy.Expr, Sign] = {}
        self._written_out_cache: dict[sy.Expr, sy.Expr] = {}
        self._new_symbols = sy.numbered_symbols(_COMPILER_INTERMEDIATE_PREFIX)

    def compile(self, steps, recipe_data=None) -> "SympyModel":
        """Compile `steps` and return the finished model."""
        for index, step in enumerate(steps):
            self.add_step(index, step)

        model = SympyModel(
            self.structure,
            values=dict(self.values),
            intermediates=self.intermediates,
            recipe_data=recipe_data,
            balance_trace=self.balance_trace(),
        )
        log_summary(model.balance_trace, _log)
        return model

    def balance_trace(self):
        """What has been found so far about each object's balance."""
        return BalanceTrace(
            self.structure,
            balances=self.written_out,
            signs=self.signs,
            events=self.events,
            incomplete=self.incomplete,
            definitions=self._definitions,
            numerical_guards=(_LIMIT_EPSILON,),
        )

    def add_step(self, index, step):
        """Resolve one step against the state so far, and accumulate it."""
        for symbol, expr, description in step.intermediates:
            self.define(symbol, self.resolve(expr), description)

        contributions = {
            symbol: self.resolve(expr) for symbol, expr in step.values.items()
        }

        # What the step would contribute if nothing held it back. A constraint
        # can only take away from a step, never add to it, so this says which
        # side of zero a balance ends up on even when the amount is uncertain.
        unconstrained = contributions if step.transformations else None
        for transform in step.transformations:
            if isinstance(transform, Limit):
                contributions = _compile_limit(contributions, transform, self)
            elif isinstance(transform, Floor):
                contributions = _compile_floor(contributions, transform, self)
            else:
                raise ValueError(f"Unknown transform {transform}")

        for i in self._objects_touched_by(contributions):
            self._record_effect(i, index, step, contributions, unconstrained)

        for symbol, expr in contributions.items():
            self.values[symbol] += expr

    def resolve(self, expr, proposed=None):
        """Replace placeholder symbols by their current accumulated values.

        As :meth:`SympyModel.resolve`, but against the state accumulated so
        far rather than a finished model.

        :param proposed: Optional ``{X[j] or Y[j]: contribution}`` to resolve
            against as though it had already been added.
        """
        return _resolve_placeholders(
            self.structure, expr, self.values, self.balances, self.signs, proposed
        )

    def mint(self, expr, description):
        """Route `expr` through a new intermediate symbol and return the symbol.

        Transformations refer to the accumulated state several times each (as
        'current', 'proposed' and the limit/threshold), and the result is
        accumulated in turn, so inlining those expressions makes each step a
        multiple of the last -- exponential in the number of steps. Hiding the
        expressions behind named intermediate symbols keeps each step's compiled
        expression a constant size. It also keeps `Piecewise` out of `Piecewise`
        conditions, which is important as sympy's ExprCondPair runs
        `piecewise_fold` and `cond.rewrite(ITE)` regardless of `evaluate=False`,
        and this is very slow for large expressions.

        Expressions that are already atomic are returned unchanged.

        """
        if not isinstance(expr, sy.Basic) or isinstance(expr, _TRIVIAL_TYPES):
            return expr
        symbol = create_intermediate(
            self.intermediates, expr, description, self._new_symbols
        )
        self._definitions[symbol] = expr
        return symbol

    def define(self, symbol, expr, description):
        """Record an intermediate symbol created by the model builder.

        These need to be expanded in order to see cancellations in the object
        balances. They are stored with earlier intermediates already substituted
        in.

        """
        expr = sy.sympify(expr)
        self.intermediates.append((symbol, expr, description))
        self._definitions[symbol] = expr
        written_out = self.write_out(expr)
        if not too_large(written_out):
            self._expansions[symbol] = written_out

    def write_out(self, expr):
        """Substitute model builder intermediate symbols in `expr`.

        Memoised: the same expressions come round repeatedly, and substituting
        into expressions containing ``Max`` is expensive.
        """
        if not isinstance(expr, sy.Basic):
            return expr
        if expr in self._written_out_cache:
            return self._written_out_cache[expr]
        symbols = {s for s in expr.free_symbols if s in self._expansions}
        if not symbols:
            result = expr
        else:
            result = expr.xreplace({s: self._expansions[s] for s in symbols})
            # xreplace on a bare symbol returns the replacement itself, which
            # need not be a sympy object if a step's value was a plain number.
            if not isinstance(result, sy.Basic):
                result = sy.S(result)
        self._written_out_cache[expr] = result
        return result

    # -- balance bookkeeping --

    def _record_effect(self, i, index, step, contributions, unconstrained):
        """Note what one step did to object `i`'s balance."""
        delta = _object_balance(self.structure, i, contributions)
        written_out = self.write_out(delta)
        if written_out == 0:
            # The step's own flows balance this object between themselves.
            return

        balance, outstanding = self.balances[i], self.written_out[i]
        sign = self.signs[i]

        if outstanding + written_out == 0:
            effect, new_sign = Effect.CLOSES, Sign.ZERO
        elif self._would_have_closed(i, outstanding, unconstrained):
            effect, new_sign = Effect.CLOSES_UNLESS_CONSTRAINED, sign
        else:
            effect = Effect.OPENS
            new_sign = combine_signs(sign, self._sign(written_out))

        self.events[i].append(
            BalanceEvent(index, effect, step.description, sign, new_sign)
        )
        closed = new_sign is Sign.ZERO
        self.balances[i] = sy.S.Zero if closed else balance + delta
        self.written_out[i] = sy.S.Zero if closed else outstanding + written_out
        self.signs[i] = new_sign
        if too_large(self.written_out[i]):
            # Stop writing this one out. The compact form is still accumulated,
            # but this object can no longer be proved to balance, and what is
            # left over is no longer a faithful account of what is missing.
            self.written_out[i] = self.balances[i]
            self.incomplete.add(i)

    def _would_have_closed(self, i, outstanding, unconstrained):
        """Whether a step would have closed object `i` had nothing held it back.

        `unconstrained` is what the step contributes before its transformations
        are applied, or None if it has none.
        """
        if unconstrained is None:
            return False
        delta = _object_balance(self.structure, i, unconstrained)
        return outstanding + self.write_out(delta) == 0

    def _objects_touched_by(self, contributions):
        """Indices of objects whose balance a set of contributions can change."""
        structure = self.structure
        touched = set()
        for symbol in contributions:
            process = structure.processes[symbol.indices[0]]
            for object_id in list(process.produces) + list(process.consumes):
                touched.add(structure.lookup_object(object_id))
        return sorted(touched)

    def _sign(self, expr):
        """:func:`~flowprog.balance.sign_of`, sharing one cache across steps."""
        return sign_of(expr, self._definitions, self._sign_cache)


class SympyModel:
    """Evaluable model with recipe data.

    This represents a complete symbolic model with associated recipe data.
    It can be evaluated repeatedly with different parameter values. The model
    is immutable after creation (though recipe can be updated if needed).

    Models are created from ModelBuilder instances via builder.build(recipe_data).
    """

    @classmethod
    def from_steps(
        cls,
        steps: list[AdditionalActivity],
        structure: ModelStructure,
        recipe_data=None,
    ) -> "SympyModel":
        """Compile a list of AdditionalActivity steps into accumulated sympy expressions.

        A thin wrapper over :class:`SympyCompiler`, which is where the work
        happens; use that directly to keep hold of the compiler afterwards.

        :param steps: list[AdditionalActivity]
        :param structure: ModelStructure (provides X, Y, S, U and connectivity)
        :returns: SympyModel
        """
        return SympyCompiler(structure).compile(steps, recipe_data)

    def __init__(
        self,
        structure: ModelStructure,
        values: dict,
        intermediates: list,
        recipe_data=None,
        balance_trace=None,
    ):
        """Initialize evaluable model (typically called via SympyCompiler).

        :param structure: Model structure (shared with builder)
        :param values: Accumulated ``{X[j] or Y[j]: expression}``
        :param intermediates: ``(symbol, expression, description)`` triples
        :param recipe_data: Recipe data (optional)
        :param balance_trace: Object market balances, as a
            :class:`~flowprog.balance.BalanceTrace`. If omitted, balance expressions
            are filled in based on `values`, but it will not be possible to identify
            which expressions are known to balance or not.
        """
        self.structure = structure

        # Frozen symbolic expressions
        self._values = defaultdict(lambda: sy.S.Zero, values)
        self._intermediates = intermediates

        if balance_trace is None:
            balance_trace = BalanceTrace(
                structure,
                balances={
                    i: _object_balance(structure, i, self._values)
                    for i in range(len(structure.objects))
                },
                definitions={sym: expr for sym, expr, _ in intermediates},
                numerical_guards=(_LIMIT_EPSILON,),
            )
        self.balance_trace = balance_trace

        # Recipe data storage
        self._recipe_by_id: dict[str, dict] = {}
        self._recipe_cache: Optional[dict[sy.Indexed, Union[float, sy.Expr]]] = None

        # Memoised expansion of the intermediates, keyed by the values it was
        # built for -- see `_expand_intermediates`.
        self._expansion_cache: Optional[tuple] = None

        if recipe_data:
            self.set_recipe(recipe_data)

    # Convenience accessors (delegate to structure)
    @property
    def processes(self):
        """Get processes from structure."""
        return self.structure.processes

    @property
    def objects(self):
        """Get objects from structure."""
        return self.structure.objects

    @property
    def X(self):
        """Get X symbols from structure."""
        return self.structure.X

    @property
    def Y(self):
        """Get Y symbols from structure."""
        return self.structure.Y

    @property
    def S(self):
        """Get S symbols from structure."""
        return self.structure.S

    @property
    def U(self):
        """Get U symbols from structure."""
        return self.structure.U

    @property
    def B(self):
        """Get B (elementary exchange) symbols from structure."""
        return self.structure.B

    def expr(
        self,
        role: str,
        *,
        process_id: Optional[str] = None,
        object_id: Optional[str] = None,
        exchange_id: Optional[str] = None,
        limit_to_processes: Optional[Container[str]] = None,
    ) -> sy.Expr:
        """Construct a symbolic expression for `role` (delegates to
        `ModelStructure.expr()`).
        """
        return self.structure.expr(
            role,
            process_id=process_id,
            object_id=object_id,
            exchange_id=exchange_id,
            limit_to_processes=limit_to_processes,
        )

    # ========== Recipe Management ==========

    def set_recipe(self, recipe_data):
        """Set recipe data for this model.

        :param recipe_data: Recipe in ontology-consistent format
            ``{process_id: {consumes: {object_id: value}, produces: {object_id: value}}}``
            OR backward-compatible symbol format
            ``{S[i, j]: value, U[i, j]: value}``
        """
        if not recipe_data:
            return

        # Detect format
        first_key = next(iter(recipe_data.keys()))

        if isinstance(first_key, sy.Indexed):
            # Symbol-based (backward compatible)
            self._set_recipe_from_symbols(recipe_data)
        else:
            # ID-based (new format)
            self._set_recipe_from_ids(recipe_data)

        # Invalidate cache. `_expansion_cache` needs no explicit invalidation:
        # it is keyed on the recipe values themselves.
        self._recipe_cache = None

    def _set_recipe_from_symbols(self, recipe_data):
        """Convert symbol-based recipe to ID-based storage."""
        self._recipe_by_id = {}

        for symbol, value in recipe_data.items():
            if not isinstance(symbol, sy.Indexed):
                continue

            if symbol.base == self.U:
                i, j = symbol.indices
                obj_id = self.objects[i].id
                proc_id = self.processes[j].id

                if proc_id not in self._recipe_by_id:
                    self._recipe_by_id[proc_id] = {
                        "consumes": {},
                        "produces": {},
                        "exchanges": {},
                    }
                self._recipe_by_id[proc_id]["consumes"][obj_id] = value

            elif symbol.base == self.S:
                i, j = symbol.indices
                obj_id = self.objects[i].id
                proc_id = self.processes[j].id

                if proc_id not in self._recipe_by_id:
                    self._recipe_by_id[proc_id] = {
                        "consumes": {},
                        "produces": {},
                        "exchanges": {},
                    }
                self._recipe_by_id[proc_id]["produces"][obj_id] = value

            elif symbol.base == self.B:
                e, j = symbol.indices
                exchange_id = self.structure.elementary_exchanges[e].id
                proc_id = self.processes[j].id

                if exchange_id not in self.processes[j].exchanges:
                    raise ValueError(
                        f"Recipe sets B value for exchange '{exchange_id}' on "
                        f"process '{proc_id}', but the process definition only "
                        f"lists: {self.processes[j].exchanges}"
                    )

                if proc_id not in self._recipe_by_id:
                    self._recipe_by_id[proc_id] = {
                        "consumes": {},
                        "produces": {},
                        "exchanges": {},
                    }
                self._recipe_by_id[proc_id]["exchanges"][exchange_id] = value

    def _set_recipe_from_ids(self, recipe_data):
        """Set recipe from ID-based format with validation."""
        for proc_id, recipe in recipe_data.items():
            # Validate process exists
            j = self.structure.lookup_process(proc_id)
            process = self.processes[j]

            # Validate consumed objects
            for obj_id in recipe.get("consumes", {}):
                if obj_id not in process.consumes:
                    raise ValueError(
                        f"Process '{proc_id}' recipe consumes '{obj_id}', "
                        f"but process definition only lists: {process.consumes}"
                    )

            # Validate produced objects
            for obj_id in recipe.get("produces", {}):
                if obj_id not in process.produces:
                    raise ValueError(
                        f"Process '{proc_id}' recipe produces '{obj_id}', "
                        f"but process definition only lists: {process.produces}"
                    )

            # Validate elementary exchanges against the process's declaration
            for exchange_id in recipe.get("exchanges", {}):
                self.structure.lookup_exchange(exchange_id)
                if exchange_id not in process.exchanges:
                    raise ValueError(
                        f"Process '{proc_id}' recipe has exchange '{exchange_id}', "
                        f"but process definition only lists: {process.exchanges}"
                    )

            # Store recipe
            self._recipe_by_id[proc_id] = recipe

    def get_recipe(self, process_id: str) -> dict:
        """Get full recipe for a process in human-readable format.

        :param process_id: Process identifier
        :return: ``{"consumes": {obj_id: value}, "produces": {obj_id: value},
            "exchanges": {exchange_id: value}}``
        """
        return self._recipe_by_id.get(
            process_id, {"consumes": {}, "produces": {}, "exchanges": {}}
        )

    def get_recipe_as_symbols(self) -> dict[sy.Indexed, Union[float, sy.Expr]]:
        """Get recipe data as symbol-based dict (cached for performance).

        :return: ``{U[i, j]: value, S[i, j]: value, B[e, j]: value, ...}``
        """
        if self._recipe_cache is None:
            self._recipe_cache = {}

            for proc_id, recipe in self._recipe_by_id.items():
                j = self.structure.lookup_process(proc_id)

                for obj_id, value in recipe.get("consumes", {}).items():
                    i = self.structure.lookup_object(obj_id)
                    self._recipe_cache[self.U[i, j]] = value

                for obj_id, value in recipe.get("produces", {}).items():
                    i = self.structure.lookup_object(obj_id)
                    self._recipe_cache[self.S[i, j]] = value

                for exchange_id, value in recipe.get("exchanges", {}).items():
                    e = self.structure.lookup_exchange(exchange_id)
                    self._recipe_cache[self.B[e, j]] = value

        return self._recipe_cache

    # ========== Evaluation Methods ==========

    def _get_value(self, symbol: sy.Indexed) -> sy.Expr:
        """Get the raw accumulated expression for a model state variable (X, Y).

        This is a low-level accessor. The value is returned as-is, without
        substituting structural placeholders, recipe symbols, or intermediate
        variables.  Use `eval()` if you want those substituted.

        :param symbol: X[j] or Y[j]
        :return: Accumulated expression (recipe/intermediate symbols
            un-substituted), or sy.S.Zero if never assigned
        """
        return self._values.get(symbol, sy.S.Zero)

    def resolve(self, expr: sy.Expr):
        """Replace structural placeholder symbols by this model's expressions.

        ``X[j]``, ``Y[j]``, ``Balance[i]``, the deficits and
        ``ElementaryBalance[e]`` mean something only relative to the accumulated
        model state. Recipe coefficients and intermediate symbols are left in
        place so the resulting expressions are small -- use :meth:`eval` to
        substitute those too, or :meth:`lambdify` to compile the result for
        repeated evaluation.

        :param expr: Expression to resolve.
        :return: Expression in terms of recipe coefficients, intermediate
            symbols and model parameters.

        """
        return _resolve_placeholders(
            self.structure,
            expr,
            self._values,
            self.balance_trace.balances,
            self.balance_trace.signs,
        )

    def _expand_intermediates(self, all_values: dict) -> dict:
        """Fully expanded definition of every intermediate symbol.

        Each intermediate is expanded exactly once, in definition order, so
        that by the time a definition is reached its own dependencies have
        already been expanded. The result is memoised against `all_values`,
        because callers such as `flowprog.reporting.evaluate_views` evaluate
        many expressions (e.g. every row of the flow table) against the same
        values, and every one of them needs the same expansion.

        Expanding the whole set of intermediates takes about as long as
        expanding it for a single expression, the memoisation is needed.

        :param all_values: Recipe and parameter values, already merged.
        :return: ``{intermediate symbol: expanded expression}``

        """
        # The key covers the recipe as well as the parameters, since
        # `all_values` has them merged -- so a changed recipe misses the memo
        # without needing `set_recipe` to invalidate it. Values may be
        # unhashable (lists, arrays), in which case just skip the memo rather
        # than failing.
        try:
            key = tuple(sorted(all_values.items(), key=lambda kv: str(kv[0])))
            hash(key)
        except TypeError:
            key = None

        if key is not None and self._expansion_cache is not None:
            cached_key, cached_expansion = self._expansion_cache
            if cached_key == key:
                return cached_expansion

        expanded: dict = {}
        for sym, sym_value, _ in self._intermediates:
            # xreplace, not subs: exact match is enough.
            value = sym_value
            if isinstance(value, sy.Expr):
                value = value.xreplace(all_values)
            # xreplace returns the raw replacement when the whole expression
            # matches, so the line above can yield a plain number.
            if not isinstance(value, sy.Basic):
                value = sy.sympify(value)
            # The second xreplace cannot reintroduce a raw value: each entry
            # of `expanded` was sympified by this same step on an earlier
            # iteration.
            expanded[sym] = value.xreplace(expanded)

        if key is not None:
            self._expansion_cache = (key, expanded)

        return expanded

    def eval_intermediates(self, expr: sy.Expr, values=None):
        """Substitute in `values` to intermediate expressions and then flows.

        :param expr: Expression to evaluate
        :param values: Additional values to substitute (recipe automatically included)
        :return: Evaluated expression
        """
        if values is None:
            values = {}

        # Merge recipe with provided values
        all_values = {**self.get_recipe_as_symbols(), **values}

        expanded = self._expand_intermediates(all_values)
        return expr.xreplace(expanded) if isinstance(expr, sy.Basic) else expr

    def eval(self, expr: sy.Expr, values=None, expand_intermediates=True):
        """Evaluate an expression against this model's accumulated state and recipe.

        Structural symbols (X[j]/Y[j], Balance[i], the deficits,
        ElementaryBalance[e], etc) are replaced by their accumulated
        expressions, and recipe symbols S[i,j]/U[i,j]/B[e,j] are replaced by
        their values. Any recipe symbol with no value available to substitute
        is preserved as-is.

        :param expr: Symbol or Sympy expression to evaluate.
        :param values: Additional parameter values to substitute.
        :param expand_intermediates: If True (default), expand the intermediate
            placeholder symbols too, yielding a self-contained expression. If
            False, leave them un-expanded. This is much faster for large models,
            and likely what you want to use for manipulating intermediate
            expressions which you will later pass to `lambdify()`. With
            `expand_intermediates=False`, `values` are substituted only into the
            visible expression (not within intermediate expressions), so
            combining the two is unusual (the point of skipping expansion is to
            keep expressions compact until a final `lambdify()`).
        :return: Evaluated expression.

        """
        # Resolve any structural symbols against final accumulated state.
        result = self.resolve(expr)
        if expand_intermediates:
            # Expands intermediates *and* substitutes recipe/values into them.
            result = self.eval_intermediates(result, values)
        if isinstance(result, sy.Basic):
            # Substitute recipe values that appeared directly (e.g. S[i,j]/
            # U[i,j] from accumulated values, or B[e,j] from an
            # elementary-flow expression). In the expanded case `values` were
            # already applied by eval_intermediates above.
            recipe_syms = self.get_recipe_as_symbols()
            if recipe_syms:
                result = result.xreplace(recipe_syms)
            # Substitute `values` into the visible expression. In the
            # expanded case eval_intermediates applied them only *inside*
            # intermediate definitions, and in the non-expanded case not at
            # all -- either way, parameters appearing directly in the
            # expression (e.g. via a recipe value B[e,j] -> EF_x -> number)
            # still need this separate pass to chain through. Any recipe
            # symbol left over (S/U/B with no value) persists, as before.
            if values and isinstance(result, sy.Basic):
                result = result.xreplace(values)
        return result

    def to_flows(self, values=None, flow_ids=None):
        """Return the technosphere flow table with values resolved.

        This is a thin wrapper over the structural `ModelStructure.flow_table`
        (the general form) resolved through this model: equivalent to
        ``evaluate_views(model, model.structure.flow_table(), values)``, with an
        optional hash-based ``id`` column. Recipe data is automatically
        included.

        If only the flow structure and ids are needed -- not the values --
        call `model.structure.flow_table(flow_ids=True)` directly, which skips
        resolving and expanding the expressions altogether.

        :param values: Additional parameter values to substitute
        :param flow_ids: If True, assign hash-based flow ids to each row
        :return: DataFrame with columns source, target, material, metric, value

        """
        from ..reporting import evaluate_views

        table = self.structure.flow_table(flow_ids=bool(flow_ids))
        return evaluate_views(self, table, values)

    def lambdify(self, data=None, expressions: Optional[dict] = None, modules=None):
        """Return function to evaluate model.

        Recipe data is automatically included (early substitution).

        :param data: Additional fixed data (beyond recipe) for early substitution
        :param expressions: Optional dict of expressions to evaluate.
                           If None, uses model flows.
        :param modules: Passed to sympy.lambdify().  Use ``'math'`` for
            scalar-only evaluation: avoids numpy array overhead and correctly
            handles nested ``Piecewise`` / ``ITE`` nodes that numpy's code
            generator can miscompile for scalar inputs.  Default ``None``
            uses numpy (suitable for vectorised / array evaluation).
        :return: Callable function that takes parameter dict and returns results
        """
        if data is None:
            data = {}

        # Merge recipe with additional data for early substitution
        all_data = {**self.get_recipe_as_symbols(), **data}

        # Default to the model's own flows, keyed by hash-based flow id --
        # the structural table, routed through the same path as any other
        # expressions dict (below) rather than eagerly evaluated.
        if expressions is None:
            flows = self.structure.flow_table()
            expressions = {
                flow_id(r): v for r, v in zip(flows.itertuples(), flows["value"])
            }

        index = list(expressions.keys())
        # Resolve structural symbols (X[j]/Y[j], Balance[i], ElementaryFlows,
        # ...) against accumulated state, so purely structural expressions --
        # the flows above, or reporting views over the structural flow
        # tables -- compile directly. Recipe symbols S/U/B and intermediates
        # are substituted later (all_data, in _lambdify), keeping expressions
        # compact and CSE-friendly.
        expr_values = [
            self.resolve(sy.S(v)) for v in expressions.values()
        ]

        # Function that returns a vector of values in same order as index
        func = self._lambdify(expr_values, all_data, modules=modules)

        # Create a friendlier wrapper
        str_args = func.__code__.co_varnames[: func.__code__.co_argcount]

        def wrapper(data):
            converted_data = convert_indexed_symbols(data)
            relevant_data = {
                str(k): v for k, v in converted_data.items() if str(k) in str_args
            }
            missing_params = set(str_args) - set(relevant_data)
            if missing_params:
                raise ValueError(f"Missing parameters: {missing_params}")
            values = func(**relevant_data)
            # Convert to float if it's a 0-dimensional array
            values = [
                float(x) if isinstance(x, np.ndarray) and x.ndim == 0 else x
                for x in values
            ]
            return dict(zip(index, values))

        return wrapper

    def _lambdify(self, values, data_for_intermediates, modules=None):
        """Internal lambdify implementation.

        :param values: Expressions to lambdify
        :param data_for_intermediates: Data to substitute into intermediates (early)
        :param modules: Passed directly to sympy.lambdify().  ``None`` uses numpy.
        :return: Lambdified function
        """
        # Substitute recipe/data in intermediates now (early substitution)
        subexpressions = [
            (
                sym,
                expr.xreplace(data_for_intermediates).xreplace(data_for_intermediates),
            )
            for sym, expr, _ in self._intermediates
        ]

        # Substitute data in values too
        values = [expr.xreplace(data_for_intermediates) for expr in values]

        # Find remaining free symbols (these become function parameters)
        args = (
            set()
            .union(*(expr.free_symbols for expr in values))
            .union(*(expr.free_symbols for _, expr in subexpressions))
            .difference(sym for sym, _ in subexpressions)
        )

        # Indexed objects return themselves (e.g. S[1, 1]) as a free symbol as
        # well as the base matrix (e.g. S) - filter these out
        args = {x for x in args if not isinstance(x, sy.Indexed)}
        args = list(args)

        kwargs = {}
        if modules is not None:
            kwargs["modules"] = modules

        f = sy.lambdify(args, values, cse=lambda expr: (subexpressions, expr), **kwargs)

        return f

    def save(self, filepath: str, metadata: Optional[dict] = None):
        """Save the complete model state including recipe to a JSON file.

        This saves all information needed to recreate the evaluable model:
        - Model structure (processes and objects)
        - All assigned values (_values)
        - Intermediate symbols and expressions (_intermediates)
        - What each object's market balances to, and how it got there
        - Recipe data (both consumes and produces)

        Symbolic expressions are serialized using SymPy's srepr() which produces
        canonical string representations that can be exactly reconstructed.

        :param filepath: Path to save the model (will be overwritten if exists)
        :param metadata: Optional dictionary of metadata to include (e.g., description, author)

        **Example**::

            model.save("my_model.json", metadata={"description": "Energy model v1.0"})
        """
        import json
        from datetime import datetime

        # Build the data structure
        data = {
            "version": "1.3",
            "metadata": metadata or {},
            "saved_at": datetime.now().isoformat(),
            "type": "SympyModel",
            "processes": [
                {
                    "id": p.id,
                    "produces": p.produces,
                    "consumes": p.consumes,
                    "has_stock": p.has_stock,
                    "exchanges": p.exchanges,
                }
                for p in self.processes
            ],
            "objects": [
                {
                    "id": o.id,
                    "metric": str(o.metric),  # Convert URIRef to string
                    "has_market": o.has_market,
                }
                for o in self.objects
            ],
            "elementary_exchanges": [
                {
                    "id": e.id,
                    "metric": str(e.metric),  # Convert URIRef to string
                }
                for e in self.structure.elementary_exchanges
            ],
            "values": {
                sy.srepr(k): sy.srepr(v)
                for k, v in self._values.items()
                if v != sy.S.Zero  # Don't save default zero values
            },
            "intermediates": [
                {
                    "symbol": sy.srepr(sym),
                    "expr": sy.srepr(expr),
                    "label": label,
                }
                for sym, expr, label in self._intermediates
            ],
            "recipe": self._recipe_by_id,  # Save the ID-based recipe
            # What each object's market comes to, and how it got there. Saved
            # because it cannot be recovered from the values alone in the same
            # form -- see SympyModel.__init__.
            "object_balances": self.balance_trace.to_dict(),
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        _log.info(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "SympyModel":
        """Load an evaluable model from a JSON file created by SympyModel.save().

        This recreates the complete evaluable model state including recipe data.

        The model is reconstructed by:
        1. Creating the model structure (processes and objects)
        2. Deserializing all symbolic expressions using sympify()
        3. Restoring values and intermediates
        4. Restoring recipe data

        Note: Due to SymPy's internal caching, IndexedBase objects (X, Y, S, U)
        created during model initialization will be automatically used by the
        deserialized expressions.

        :param filepath: Path to the saved model file
        :return: Reconstructed Model instance (evaluable)

        **Example**::

            model = SympyModel.load("my_model.json")
        """
        import json

        with open(filepath) as f:
            data = json.load(f)

        # Check version compatibility
        if data.get("version") not in ("1.0", "1.1", "1.2", "1.3"):
            _log.warning(
                f"Model file version {data.get('version')} may not be compatible "
                "with this version of flowprog"
            )

        # Check if this is an evaluable model or builder
        if (data_type := data.get("type")) != "SympyModel":
            _log.warning(
                f"SympyModel.load trying to load data with type = '{data_type}'"
            )

        # Reconstruct processes and objects
        processes = [
            Process(
                id=p["id"],
                produces=p["produces"],
                consumes=p["consumes"],
                has_stock=p["has_stock"],
                exchanges=p.get("exchanges", []),
            )
            for p in data["processes"]
        ]

        # Files saved before exchange sparsity became structural (< v1.2)
        # have no per-process declarations; backfill them from the recipe so
        # set_recipe() validation below still passes.
        saved_recipe = data.get("recipe") or {}
        for process, saved in zip(processes, data["processes"]):
            if "exchanges" not in saved:
                process.exchanges = list(
                    saved_recipe.get(process.id, {}).get("exchanges", {})
                )

        objects = [
            Object(
                id=o["id"],
                metric=URIRef(o["metric"]),  # Convert string back to URIRef
                has_market=o["has_market"],
            )
            for o in data["objects"]
        ]

        elementary_exchanges = [
            ElementaryExchange(
                id=e["id"],
                metric=URIRef(e["metric"]),  # Convert string back to URIRef
            )
            for e in data.get("elementary_exchanges", [])
        ]

        # Create structure
        structure = ModelStructure(processes, objects, elementary_exchanges)

        # Prepare namespace for sympify
        namespace = {
            "X": structure.X,
            "Y": structure.Y,
            "S": structure.S,
            "U": structure.U,
            "B": structure.B,
            "ElementaryBalance": structure.ElementaryBalance,
        }

        # Restore values
        values = defaultdict(lambda: sy.S.Zero)
        for k_str, v_str in data["values"].items():
            try:
                k = sy.sympify(k_str, locals=namespace)
                v = sy.sympify(v_str, locals=namespace)
                values[k] = v
            except Exception as e:
                _log.error(f"Failed to deserialize value {k_str}: {e}")
                raise

        # Restore intermediates
        intermediates = []
        for item in data["intermediates"]:
            try:
                sym = sy.sympify(item["symbol"], locals=namespace)
                expr = sy.sympify(item["expr"], locals=namespace)
                intermediates.append((sym, expr, item["label"]))
            except Exception as e:
                _log.error(f"Failed to deserialize intermediate {item['symbol']}: {e}")
                raise

        # Restore what was recorded about the object balances. Absent from
        # files written before this was tracked, in which case the model works
        # them out from the values instead.
        saved_balances = data.get("object_balances")
        balance_trace = (
            BalanceTrace.from_dict(
                saved_balances,
                structure,
                definitions={sym: expr for sym, expr, _ in intermediates},
                numerical_guards=(_LIMIT_EPSILON,),
                sympify=lambda expr: sy.sympify(expr, locals=namespace),
            )
            if saved_balances
            else None
        )

        # Create model with recipe
        model = cls(
            structure=structure,
            values=values,
            intermediates=intermediates,
            recipe_data=data.get("recipe"),  # Restore recipe
            balance_trace=balance_trace,
        )

        _log.info(f"Model loaded from {filepath}")
        if "metadata" in data and data["metadata"]:
            _log.info(f"Metadata: {data['metadata']}")

        return model


def _compile_limit(contributions, limit, state):
    """Compile a Limit transformation into Piecewise expressions.

    A limit constrains an expression not to exceed a bound, by scaling down the
    step it applies to. Three values decide by how much:

    - *current*: what the expression comes to as things stand;
    - *proposed*: what it would come to if this step were added in full;
    - *bound*: the limit itself, as things stand.

    Each of those, and the step's own contributions, are routed through
    intermediate symbols: they appear several times below, and naming them
    keeps each step's compiled expression a constant size.
    """
    contributions = {
        symbol: state.mint(expr, f"limit input for {symbol}")
        for symbol, expr in contributions.items()
    }

    current = state.mint(
        state.resolve(limit.expression),
        f"limit: current value of {limit.expression}",
    )
    bound = state.mint(
        state.resolve(limit.limit_value),
        f"limit: bound {limit.limit_value}",
    )
    proposed = state.mint(
        state.resolve(limit.expression, proposed=contributions),
        f"limit: proposed value of {limit.expression}",
    )

    # `proposed` and `current` name the same subexpressions the inlined form
    # would have evaluated, so this subtracts the same two floats as before,
    # but keeps `safe_difference` a constant size -- which matters because
    # sy.Max() runs an `equals()` comparison over its arguments that is very
    # slow on anything Piecewise-laden.
    safe_difference = sy.Max(proposed - current, _LIMIT_EPSILON)

    return {
        symbol: sy.Piecewise(
            (sy.S.Zero, current >= bound),
            (expr, proposed <= bound),
            ((bound - current) / safe_difference * expr, True),
            evaluate=False,
        )
        for symbol, expr in contributions.items()
    }


def _compile_floor(contributions, floor, state):
    """Compile a Floor transformation into Piecewise expressions.

    A floor is a minimum operating level: if adding this step in full would
    still leave the expression below the threshold, the step is dropped
    entirely rather than run below the minimum.
    """
    contributions = {
        symbol: state.mint(expr, f"floor input for {symbol}")
        for symbol, expr in contributions.items()
    }

    proposed = state.mint(
        state.resolve(floor.expression, proposed=contributions),
        f"floor: proposed value of {floor.expression}",
    )
    threshold = state.mint(
        state.resolve(floor.threshold),
        f"floor: threshold {floor.threshold}",
    )

    return {
        symbol: sy.Piecewise(
            (expr, proposed >= threshold),
            (sy.S.Zero, True),
            evaluate=False,
        )
        for symbol, expr in contributions.items()
    }


def convert_indexed_symbols(data):
    """Convert {S[1, 2]: 7} to {S: {(1, 2): 7}}

    This works better with lambdified functions.
    """
    converted = {}
    for k, v in data.items():
        if isinstance(k, sy.Indexed):
            sym = k.base
            if sym not in converted:
                converted[sym] = {}
            indices = k.indices if len(k.indices) > 1 else k.indices[0]
            converted[sym][indices] = v
        else:
            converted[k] = v
    return converted
