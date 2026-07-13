"""Reporting: a thin, pure aggregation layer over evaluated flow tables.

Every report is a group-by over disjoint cells of the (exchange x process)
elementary-flow matrix, so double counting is structurally impossible
provided groupings are partitions. Characterisation (e.g. GWP) is user data,
applied at aggregation time, never mixed into the underlying B rows.

Model-specific declarations (the actual stage map, EF symbols, named totals)
belong in the model repository, not here -- this module only supplies the
generic machinery.
"""

import logging
from typing import Iterable, Optional, Union

import pandas as pd
import sympy as sy

_log = logging.getLogger(__name__)


class Grouping:
    """Validated id -> label mapping, with an explicit "other" bucket.

    Groups must be disjoint (no id assigned to two labels); any id from
    `all_ids` not covered by a group falls into `other_label`, logging a
    warning listing which ids fell through.
    """

    @staticmethod
    def build(
        name: str,
        groups: dict[str, Iterable[str]],
        all_ids: Iterable[str],
        other_label: str = "other",
    ) -> dict[str, str]:
        """Build a validated id -> label mapping.

        :param name: Grouping name, used only in error/warning messages.
        :param groups: label -> ids in that group. Must be disjoint. Declaring
            ``other_label`` here (even as an empty set) marks the remainder
            bucket as intentional -- ids not covered by a group then fall into
            it *silently*. Omit it and any remainder is bucketed into
            ``other_label`` with a warning, so a forgotten id is surfaced.
        :param all_ids: The full universe of ids this grouping should cover.
        :param other_label: Label assigned to any id not covered by a group.
        :return: id -> label mapping, including `other_label` entries.
        :raises ValueError: If groups overlap, or reference an id outside `all_ids`.
        """
        all_ids = set(all_ids)
        mapping: dict[str, str] = {}

        for label, ids in groups.items():
            for id_ in ids:
                if id_ not in all_ids:
                    raise ValueError(
                        f"Grouping {name!r}: group {label!r} references unknown id {id_!r}"
                    )
                if id_ in mapping:
                    raise ValueError(
                        f"Grouping {name!r}: id {id_!r} assigned to multiple groups "
                        f"({mapping[id_]!r} and {label!r}) -- groups must be disjoint"
                    )
                mapping[id_] = label

        remainder = all_ids - set(mapping)
        # Warn only when the remainder is unexpected -- i.e. the caller did not
        # declare an explicit catch-all group. An explicit (possibly empty)
        # `other_label` group signals "everything else is intentionally here".
        if remainder and other_label not in groups:
            _log.warning(
                "Grouping %r: %d id(s) not covered by any group, assigned to %r: %s",
                name,
                len(remainder),
                other_label,
                sorted(remainder),
            )
        for id_ in remainder:
            mapping[id_] = other_label

        return mapping


def _infer_axis(name, groups, process_ids, exchange_ids):
    all_group_ids = {id_ for ids in groups.values() for id_ in ids}
    if all_group_ids <= process_ids:
        return "process", process_ids
    if all_group_ids <= exchange_ids:
        return "exchange", exchange_ids
    unknown = all_group_ids - process_ids - exchange_ids
    raise ValueError(
        f"Grouping {name!r}: ids are neither all process ids nor all exchange "
        f"ids of the model (unrecognised: {sorted(unknown) or sorted(all_group_ids)})"
    )


class Reporting:
    """Pure post-processing over an evaluated model's elementary-flow table.

    :param model: A built, evaluable model (e.g. SympyModel)
    :param groupings: ``{grouping_name: {label: {ids}}}`` -- ids may be
        process ids or elementary exchange ids (inferred per grouping); a
        grouping's ids must be entirely one or the other.
    :param characterisations: ``{characterisation_name: {exchange_id: factor}}``.
        An exchange id absent from a characterisation's factors contributes
        zero to that characterisation.
    :param pass_through: Optional `flowprog.allocation.PassThrough` (or an
        iterable of process ids, from which one is constructed). If given,
        all aggregation runs over the *reattributed* elementary-flow table --
        each pass-through process's burden redistributed to its direct
        consumers -- instead of the model's own table.
    :param lambdify_modules: Passed to the model's ``lambdify()`` when
        evaluating with ``values=``. Use ``"math"`` for large models with
        nested Piecewise expressions (see ``SympyModel.lambdify``).

    **Example**::

        rep = Reporting(model, groupings={"stage": stage_map}, characterisations={"GWP": gwp})
        rep.aggregate("GWP")                          # scalar (symbolic)
        rep.aggregate("GWP", by="stage", values=data) # Series by stage (numeric)
    """

    def __init__(
        self,
        model,
        groupings: Optional[dict[str, dict[str, Iterable[str]]]] = None,
        characterisations: Optional[dict[str, dict[str, float]]] = None,
        pass_through=None,
        lambdify_modules=None,
    ):
        self.model = model
        self.characterisations = dict(characterisations or {})
        self._lambdify_modules = lambdify_modules

        if pass_through is not None and not hasattr(pass_through, "elementary_flows"):
            from .allocation import PassThrough

            pass_through = PassThrough(model, pass_through)
        self.pass_through = pass_through

        process_ids = {p.id for p in model.processes}
        exchange_ids = {e.id for e in model.structure.elementary_exchanges}

        self._groupings: dict[str, tuple[str, dict[str, str]]] = {}
        for name, groups in (groupings or {}).items():
            axis, all_ids = _infer_axis(name, groups, process_ids, exchange_ids)
            self._groupings[name] = (axis, Grouping.build(name, groups, all_ids))

        self._named_totals: dict[str, dict] = {}
        self._flows_cache: Optional[pd.DataFrame] = None
        self._technosphere_cache: dict[tuple[str, str], pd.DataFrame] = {}
        self._lambdify_cache: dict = {}

    def _flows_table(self) -> pd.DataFrame:
        """The (raw, unresolved) elementary-flow table all aggregation runs over.

        Values are raw symbolic expressions (intermediate placeholders left
        unresolved -- see ``SympyModel.to_elementary_flows(raw=True)``), so
        building the table is cheap even for large models; resolution happens
        once per query, via ``eval()`` (symbolic results) or the model's
        lambdify fast path (``values=`` results).
        """
        if self._flows_cache is None:
            if self.pass_through is not None:
                self._flows_cache = self.pass_through.elementary_flows()
            else:
                self._flows_cache = self.model.to_elementary_flows(raw=True)
        return self._flows_cache

    def _column_values(self, key: str, table: pd.DataFrame) -> list:
        if key == "process":
            return list(table["process"])
        if key == "exchange":
            return list(table["exchange"])
        if key in self._groupings:
            axis, mapping = self._groupings[key]
            col = "process" if axis == "process" else "exchange"
            return [mapping[v] for v in table[col]]
        raise ValueError(
            f"Unknown grouping or column {key!r}; declared groupings: "
            f"{sorted(self._groupings)}"
        )

    def _get_lambdified(self, cache_key, build_exprs):
        if cache_key not in self._lambdify_cache:
            self._lambdify_cache[cache_key] = self.model.lambdify(
                expressions=build_exprs(), modules=self._lambdify_modules
            )
        return self._lambdify_cache[cache_key]

    def _resolve(self, expr):
        """Resolve a raw aggregate expression to fully-substituted symbolic form."""
        return self.model.eval(expr)

    def aggregate(
        self,
        characterisation: Optional[str],
        by: Optional[Union[str, tuple]] = None,
        limit_to_processes: Optional[Iterable[str]] = None,
        values: Optional[dict] = None,
        raw: bool = False,
    ):
        """Aggregate elementary flows, optionally characterised and grouped.

        :param characterisation: Name of a declared characterisation vector to
            apply, or None to sum raw (uncharacterised) flow values.
        :param by: None for a scalar total; a grouping/column name (or "process"
            /"exchange" for the raw ids) for a Series; a tuple of names for a
            MultiIndex Series.
        :param limit_to_processes: Optional set of process ids to restrict to.
        :param values: Optional parameter values. If given, evaluates via the
            model's lambdify fast path (compiled function is cached per
            (characterisation, by, limit_to_processes) so repeated calls with
            different values -- e.g. in a Monte Carlo loop -- are cheap).
        :param raw: Only meaningful with ``values=None``. If True, return
            *raw* symbolic expressions (recipe values substituted but
            intermediate placeholder symbols left unresolved) instead of
            fully-substituted ones. Raw expressions are cheap to build for
            large models and are exactly what the model's
            ``lambdify(expressions=...)`` expects, so use this to merge
            aggregates into a wider results dict compiled in one pass.
        :return: A sympy expression (scalar, values=None), a float (scalar,
            values given), or a pandas Series (by given).
        """
        table = self._flows_table()

        if limit_to_processes is not None:
            allowed = set(limit_to_processes)
            mask = [p in allowed for p in table["process"]]
        else:
            mask = None

        if characterisation is None:
            factors = None
        else:
            factors = self.characterisations[characterisation]

        def row_value(idx):
            v = table["value"].iat[idx]
            if factors is None:
                return v
            e = table["exchange"].iat[idx]
            return v * factors.get(e, 0)

        indices = [
            i for i in range(len(table)) if mask is None or mask[i]
        ]

        def scalar_total():
            return sum((row_value(i) for i in indices), sy.S.Zero)

        if by is None:
            cache_key = (
                "scalar",
                characterisation,
                _freeze(limit_to_processes),
            )
            if values is None:
                total = scalar_total()
                return total if raw else self._resolve(total)
            func = self._get_lambdified(cache_key, lambda: {"total": scalar_total()})
            return func(values)["total"]

        by_keys = (by,) if isinstance(by, str) else tuple(by)
        group_cols = [self._column_values(k, table) for k in by_keys]

        def build_grouped():
            grouped: dict = {}
            for i in indices:
                key = (
                    group_cols[0][i]
                    if len(group_cols) == 1
                    else tuple(col[i] for col in group_cols)
                )
                grouped.setdefault(key, []).append(row_value(i))
            return {k: sum(vs, sy.S.Zero) for k, vs in grouped.items()}

        if values is None:
            result_dict = build_grouped()
            if not raw:
                result_dict = {k: self._resolve(v) for k, v in result_dict.items()}
        else:
            cache_key = ("grouped", characterisation, by_keys, _freeze(limit_to_processes))
            func = self._get_lambdified(cache_key, build_grouped)
            result_dict = func(values)

        if len(by_keys) > 1:
            index = pd.MultiIndex.from_tuples(list(result_dict.keys()), names=by_keys)
            return pd.Series(list(result_dict.values()), index=index)
        return pd.Series(result_dict).rename_axis(by_keys[0])

    def _technosphere_table(self, flow_role: str, object_id: str) -> pd.DataFrame:
        """Raw per-process breakdown of one object's production or consumption.

        :param flow_role: "ProcessOutput" (Y[j]*S[i,j], one row per producer)
            or "ProcessInput" (X[j]*U[i,j], one row per consumer).
        """
        cache = self._technosphere_cache.setdefault(flow_role, {})
        if object_id not in cache:
            pids = (
                self.model.structure.producers_of(object_id)
                if flow_role == "ProcessOutput"
                else self.model.structure.consumers_of(object_id)
            )
            values = [
                self.model.eval(
                    self.model.expr(flow_role, process_id=pid, object_id=object_id),
                    expand_intermediates=False,
                )
                for pid in pids
            ]
            cache[object_id] = pd.DataFrame({"process": pids, "value": values})
        return cache[object_id]

    def _technosphere_aggregate(
        self,
        flow_role: str,
        object_id: str,
        by: Optional[str],
        limit_to_processes: Optional[Iterable[str]],
        values: Optional[dict],
        raw: bool,
    ):
        if by is not None and by != "process":
            axis = self._groupings.get(by, (None,))[0]
            if axis != "process":
                raise ValueError(
                    f"{by!r} is not a process grouping; production()/consumption() "
                    "can only be grouped by process id or a process-axis grouping"
                )

        table = self._technosphere_table(flow_role, object_id)

        if limit_to_processes is not None:
            allowed = set(limit_to_processes)
            mask = [p in allowed for p in table["process"]]
        else:
            mask = None

        indices = [i for i in range(len(table)) if mask is None or mask[i]]

        def scalar_total():
            return sum((table["value"].iat[i] for i in indices), sy.S.Zero)

        cache_prefix = (flow_role, object_id)

        if by is None:
            cache_key = ("scalar", cache_prefix, _freeze(limit_to_processes))
            if values is None:
                total = scalar_total()
                return total if raw else self._resolve(total)
            func = self._get_lambdified(cache_key, lambda: {"total": scalar_total()})
            return func(values)["total"]

        group_col = self._column_values(by, table)

        def build_grouped():
            grouped: dict = {}
            for i in indices:
                grouped.setdefault(group_col[i], []).append(table["value"].iat[i])
            return {k: sum(vs, sy.S.Zero) for k, vs in grouped.items()}

        if values is None:
            result_dict = build_grouped()
            if not raw:
                result_dict = {k: self._resolve(v) for k, v in result_dict.items()}
        else:
            cache_key = ("grouped", cache_prefix, by, _freeze(limit_to_processes))
            func = self._get_lambdified(cache_key, build_grouped)
            result_dict = func(values)

        return pd.Series(result_dict).rename_axis(by)

    def production(
        self,
        object_id: str,
        by: Optional[str] = None,
        limit_to_processes: Optional[Iterable[str]] = None,
        values: Optional[dict] = None,
        raw: bool = False,
    ):
        """Aggregate an object's production (`Sum_j Y[j]*S[i,j]`) across its producers.

        The technosphere analogue of `aggregate()`: a group-by over
        (process, object) cells instead of (process, exchange) cells. Useful
        for reporting totals that are really a group-by over consumption/
        production rather than an ad-hoc per-process symbol (e.g. utility
        requirements) -- see this class's docstring.

        :param object_id: Object to report production of.
        :param by: None for a scalar total; "process" for a Series by raw
            process id; or the name of a declared process-axis grouping
            (e.g. "stage").
        :param limit_to_processes: Optional set of process ids to restrict to.
        :param values: Optional parameter values (see `aggregate()`).
        :param raw: See `aggregate()`.
        """
        return self._technosphere_aggregate(
            "ProcessOutput", object_id, by, limit_to_processes, values, raw
        )

    def consumption(
        self,
        object_id: str,
        by: Optional[str] = None,
        limit_to_processes: Optional[Iterable[str]] = None,
        values: Optional[dict] = None,
        raw: bool = False,
    ):
        """Aggregate an object's consumption (`Sum_j X[j]*U[i,j]`) across its consumers.

        See `production()` for parameters -- the same, mirrored for the
        consumption side (`expr("Consumption", ...)`'s group-by analogue).
        """
        return self._technosphere_aggregate(
            "ProcessInput", object_id, by, limit_to_processes, values, raw
        )

    def table(
        self,
        values: Optional[dict] = None,
        limit_to_processes: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:
        """Return a tidy DataFrame of elementary flows plus grouping columns.

        :param values: Optional parameter values to substitute (evaluates
            immediately, no lambdify caching -- use `aggregate()` for the fast
            path in tight loops).
        :param limit_to_processes: Optional set of process ids to restrict to.
        """
        table = self._flows_table().copy()
        table["value"] = [self.model.eval(v, values) for v in table["value"]]
        for name in self._groupings:
            table[name] = self._column_values(name, table)
        if limit_to_processes is not None:
            allowed = set(limit_to_processes)
            table = table[table["process"].isin(allowed)].reset_index(drop=True)
        return table

    def define_total(
        self,
        name: str,
        characterisation: Optional[str] = None,
        by: Optional[Union[str, tuple]] = None,
        limit_to_processes: Optional[Iterable[str]] = None,
    ):
        """Save a (characterisation, by, limit_to_processes) query as a named total.

        :param name: Name under which to save this query
        """
        self._named_totals[name] = dict(
            characterisation=characterisation,
            by=by,
            limit_to_processes=limit_to_processes,
        )

    def total(self, name: str, values: Optional[dict] = None, raw: bool = False):
        """Evaluate a single named total defined via `define_total()`."""
        return self.aggregate(**self._named_totals[name], values=values, raw=raw)

    def totals(self, values: Optional[dict] = None, raw: bool = False) -> dict:
        """Evaluate all named totals defined via `define_total()`."""
        return {
            name: self.total(name, values=values, raw=raw)
            for name in self._named_totals
        }


def _freeze(value):
    """Make a value hashable for use as (part of) a cache key."""
    if value is None:
        return None
    return tuple(sorted(value))
