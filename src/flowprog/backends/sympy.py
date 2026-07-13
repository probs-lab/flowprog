"""SymPy compiler: compiles model steps into accumulated symbolic expressions.

This is the default compiler used by ModelBuilder._compile().
"""

import hashlib
import logging
from typing import Container, Optional, Union
from collections import defaultdict
import sympy as sy
import numpy as np
import pandas as pd
from rdflib import URIRef

from ..model_structure import ModelStructure, Process, Object, ElementaryExchange
from ..activities import AdditionalActivity, Limit, Floor

_log = logging.getLogger(__name__)


class SympyModel:
    """Evaluable model with recipe data.

    Model represents a complete symbolic model with associated recipe data.
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

        Walks steps in order, resolving structural symbols (X[j], Y[j],
        Balance[i], ProductionDeficit[i], ConsumptionDeficit[i]) against
        accumulated state and applying transformations.

        :param steps: list[AdditionalActivity]
        :param structure: ModelStructure (provides X, Y, S, U and connectivity)
        :returns: SympyModel
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
                resolved_values[sym] = structure.resolve_structural_symbols(
                    expr, values
                )

            # 3. Apply transformations
            for transform in step.transformations:
                if isinstance(transform, Limit):
                    resolved_values = _compile_limit(
                        resolved_values, transform, values, structure
                    )
                elif isinstance(transform, Floor):
                    resolved_values = _compile_floor(
                        resolved_values, transform, values, structure
                    )
                else:
                    raise ValueError(f"Unknown transform {transform}")

            # 4. Accumulate
            for sym, expr in resolved_values.items():
                values[sym] += expr

        values = dict(values)
        return cls(structure, values, all_intermediates, recipe_data)

    def __init__(
        self,
        structure: ModelStructure,
        values: dict,
        intermediates: list,
        recipe_data=None,
    ):
        """Initialize evaluable model (typically called from SympyModel.from_steps()).

        :param structure: Model structure (shared with builder)
        :param values: Symbolic expressions from builder
        :param intermediates: Intermediate symbols from builder
        :param recipe_data: Recipe data (optional)
        """
        self.structure = structure

        # Frozen symbolic expressions
        self._values = defaultdict(lambda: sy.S.Zero, values)
        self._intermediates = intermediates

        # Recipe data storage
        self._recipe_by_id: dict[str, dict] = {}
        self._recipe_cache: Optional[dict[sy.Indexed, Union[float, sy.Expr]]] = None

        if recipe_data:
            self.set_recipe(recipe_data)

    def __repr__(self):
        return f"SympyModel(structure={self.structure}, has_recipe={bool(self._recipe_by_id)})"

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

        # Invalidate cache
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

            # Validate elementary exchanges are declared on the model
            for exchange_id in recipe.get("exchanges", {}):
                self.structure.lookup_exchange(exchange_id)

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

        intermediates = [
            (
                sym,
                (
                    sym_value.xreplace(all_values)
                    if isinstance(sym_value, sy.Expr)
                    else sym_value
                ),
            )
            for sym, sym_value, _ in self._intermediates
        ]
        return expr.subs(intermediates[::-1])

    def eval(self, expr: sy.Expr, values=None, expand_intermediates=True):
        """Evaluate an expression against this model's accumulated state and recipe.

        Structural symbols (X[j]/Y[j], Balance[i], the deficits,
        ElementaryBalance[e], etc) are replaced by their accumulated
        expressions, and recipe symbols are replaced by their (U/S/B) by their
        values. Elementary exchanges B[e,j] with no recipe entry are assumed to
        be zero and omitted from the result; recipe symbols S[i,j] and U[i,j] in
        contrast are preserved if they have no value available to substitute,
        since the structure of the system is known but the list of elementary
        exchanges per process is not explicitly defined.

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
        result = self.structure.resolve_structural_symbols(expr, self._values)
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
            # Intermediates weren't expanded here, so `values` (which target
            # intermediate definitions) weren't applied; substitute them into
            # the visible expression -- a separate pass so a recipe value that
            # is itself a parameter (e.g. B[e,j] -> EF_x -> number) chains.
            if not expand_intermediates and values and isinstance(result, sy.Basic):
                result = result.xreplace(values)
            # xreplace() can collapse a bare Indexed expression (e.g.
            # eval(S[i,j])) into a raw Python value when the whole expression
            # matches a substitution key -- re-check before the sympy-only call.
            if isinstance(result, sy.Basic):
                # A B[e,j] left over here has no recipe entry for that
                # (exchange, process) pair, which means process j simply has
                # no such exchange -- implicitly zero.
                remaining_B = {
                    a: sy.S.Zero for a in result.atoms(sy.Indexed) if a.base == self.B
                }
                if remaining_B:
                    result = result.xreplace(remaining_B)
        return result

    def to_flows(self, values=None, flow_ids=None):
        """Return flows data frame with variables substituted.

        Recipe data is automatically included.

        :param values: Additional parameter values to substitute
        :param flow_ids: If True, assign hash-based flow ids to each row
        :return: DataFrame with flows
        """
        if values is None:
            values = {}

        rows = []
        M = len(self.processes)

        for j in range(M):
            for i in (
                self.structure._obj_name_to_idx[name]
                for name in self.processes[j].produces
            ):
                expr = self._values[self.Y[j]] * self.S[i, j]
                # Resolve intermediates first, then substitute values
                resolved = self.eval(expr, values)
                rows.append(
                    (
                        self.processes[j].id,
                        self.objects[i].id,
                        self.objects[i].id,
                        self.objects[i].metric,
                        resolved,
                    )
                )

        for j in range(M):
            for i in (
                self.structure._obj_name_to_idx[name]
                for name in self.processes[j].consumes
            ):
                expr = self._values[self.X[j]] * self.U[i, j]
                # Resolve intermediates first, then substitute values
                resolved = self.eval(expr, values)
                rows.append(
                    (
                        self.objects[i].id,
                        self.processes[j].id,
                        self.objects[i].id,
                        self.objects[i].metric,
                        resolved,
                    )
                )

        result = pd.DataFrame(
            rows, columns=["source", "target", "material", "metric", "value"]
        )

        if flow_ids:

            def flow_id(row):
                return hashlib.md5(
                    (row["source"] + row["target"] + row["material"]).encode()
                ).hexdigest()

            result["id"] = result.apply(flow_id, axis=1)

        return result

    def to_elementary_flows(self, values=None, raw=False):
        """Return a tidy table of elementary exchange flows (B[e,j] * Y[j]).

        The result never includes unknown `B[e,j]` symbols: only (exchange,
        process) where `B` values are known are included.

        :param values: Additional parameter values to substitute.
        :param raw: If True, each value is the raw accumulated Y expression
            times the recipe B value, with intermediate placeholder symbols left
            unresolved. This is much faster for large models (no per-row
            `eval()`); resolve the expressions afterwards in one pass via
            `lambdify(expressions=...)` or `eval()`. Cannot be combined with
            `values`.
        :return: DataFrame with columns exchange, process, metric, value

        """

        # FIXME: should `raw` here be the same as `expand_intermediates=False`?

        if raw and values is not None:
            raise ValueError("`values` cannot be combined with raw=True")
        if values is None:
            values = {}

        rows = []
        for proc_id, recipe in self._recipe_by_id.items():
            j = self.structure.lookup_process(proc_id)
            for exchange_id, b_value in recipe.get("exchanges", {}).items():
                e = self.structure.lookup_exchange(exchange_id)
                if raw:
                    resolved = self._values[self.Y[j]] * sy.S(b_value)
                else:
                    expr = self._values[self.Y[j]] * self.B[e, j]
                    resolved = self.eval(expr, values)
                rows.append(
                    (
                        exchange_id,
                        proc_id,
                        self.structure.elementary_exchanges[e].metric,
                        resolved,
                    )
                )

        return pd.DataFrame(rows, columns=["exchange", "process", "metric", "value"])

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

        if expressions is None:
            flows_sym = self.to_flows(all_data, flow_ids=True)
            index = flows_sym["id"]
            expr_values = flows_sym.value.values
        else:
            index = expressions.keys()
            expr_values = expressions.values()

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
            "version": "1.1",
            "metadata": metadata or {},
            "saved_at": datetime.now().isoformat(),
            "type": "SympyModel",
            "processes": [
                {
                    "id": p.id,
                    "produces": p.produces,
                    "consumes": p.consumes,
                    "has_stock": p.has_stock,
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
        if data.get("version") not in ("1.0", "1.1"):
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
            )
            for p in data["processes"]
        ]

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

        # Create model with recipe
        model = cls(
            structure=structure,
            values=values,
            intermediates=intermediates,
            recipe_data=data.get("recipe"),  # Restore recipe
        )

        _log.info(f"Model loaded from {filepath}")
        if "metadata" in data and data["metadata"]:
            _log.info(f"Metadata: {data['metadata']}")

        return model


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


def _compile_floor(values_dict, floor_transform, accumulated_values, structure):
    """Compile a Floor transformation into Piecewise expressions.

    The Floor's expression and threshold contain raw structural symbols.
    The compiler resolves them to determine:
    - 'proposed': expression value after adding this step's contribution
    - 'threshold': the minimum acceptable value

    If proposed >= threshold, the step values pass through unchanged.
    Otherwise they are zeroed out (the process doesn't operate at all).
    """
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
        floor_transform.expression, proposed_values
    )

    # 'threshold' resolved against accumulated state
    threshold_resolved = structure.resolve_structural_symbols(
        floor_transform.threshold, accumulated_values
    )

    return {
        k: sy.Piecewise(
            (v, proposed >= threshold_resolved),
            (sy.S.Zero, True),
            evaluate=False,
        )
        for k, v in values_dict.items()
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
