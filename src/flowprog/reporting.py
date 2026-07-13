"""Reporting: composable symbolic views over tidy flow tables.

This works in three stages:

1. **Flow tables** -- a tidy DataFrame whose ``value`` column holds purely
   structural symbolic expressions (e.g. ``Y[j] * B[e, j]`` for elementary
   exchanges, or ``Y[j] * S[i, j]`` for normal flows), one row per declared
   (exchange, process) cell -- see :meth:`flowprog.ModelStructure.flow_table`
   and :meth:`flowprog.ModelStructure.elementary_flow_table``. These tables
   depend only on the model structure, no logic, or recipe values are yet
   included. If needed, the flow tables can be modified e.g. by
   :meth:`flowprog.allocation.PassThrough.elementary_flows`.

2. **Symbolic views** -- :class:`Report` wraps such a table and derives
   aggregate sympy expressions from it: add grouping label columns
   (:meth:`Report.with_group`), apply a characterisation vector
   (:meth:`Report.characterise`), restrict rows (:meth:`Report.filter`), then
   reduce with :meth:`Report.total` (a scalar expression) or :meth:`Report.by`
   (a Series of expressions). Everything stays symbolic; grouping is a plain
   pandas group-by over label columns, and the underlying table is always
   available as ``.table`` for anything these helpers don't cover.

3. **Evaluation** -- the model enters only here, to provide values for the
   structural symbols. Resolve one-off by substitution (:func:`evaluate_views`
   with a ``SympyModel``), compile once into a fast numeric function for
   repeated evaluation (:func:`lambdify_views`), or manipulate further with
   sympy as required.

Because every report is a group-by over disjoint cells of the (exchange x
process) elementary-flow matrix, or the (object x process) technosphere-flow
matrix, double counting is structurally impossible provided groupings are
partitions -- :class:`Grouping` validates this. Characterisation (e.g. GWP) is
user data applied at view-building time, never mixed into the underlying B rows.
Summing *different* exchange types without a characterisation is conceptually a
characterisation too (a vector of ones), so
:meth:`Report.total`/:meth:`Report.by` refuse to collapse distinct exchanges
unless a characterisation has been applied -- use ``characterise(1)`` to opt in
explicitly.

**Example**::

    rep = Report.elementary_flows(model.structure).with_group("stage", stages, on="process")
    gwp = rep.characterise(GWP, name="GWP")
    views = {
        "GWP_total": gwp.total(),
        "GWP_by_stage": gwp.by("stage"),
    }
    evaluate_views(model, views, {demand: 100})  # one-off, by substitution
    f = lambdify_views(model, views)             # compile once...
    f({demand: 100})                             # ...evaluate many times

"""

import logging
from collections.abc import Iterable, Mapping
from typing import Optional

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
        :param groups: label -> ids in that group. Must be disjoint. Any id
            from `all_ids` not covered is bucketed into ``other_label`` with
            a warning, so a forgotten id is flagged; to have an intentional
            catch-all without the warning, list its ids explicitly (e.g.
            via `complete()`).
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
        if remainder:
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

    @staticmethod
    def complete(
        groups: dict[str, Iterable[str]],
        all_ids: Iterable[str],
        other_label: str = "other",
    ) -> dict[str, set]:
        """Complete `groups` into a partition of `all_ids` by adding an
        explicit catch-all group of all uncovered ids.

        `build()` warns about any ids left uncovered (a typo guard); an
        *intentional* remainder bucket instead lists its ids explicitly --
        this helper computes that list.

        :param groups: label -> ids, as for `build()`.
        :param all_ids: The full universe of ids the grouping should cover.
        :param other_label: Label for the added catch-all group.
        :return: `groups` plus an `other_label` group of the uncovered ids.
        """
        named = set().union(*groups.values()) if groups else set()
        return {**groups, other_label: set(all_ids) - named}


class Report:
    """Immutable symbolic view builder over a tidy table of structural expressions.

    Wraps a DataFrame with (at least) a ``value`` column of sympy
    expressions; for elementary flows the table also has ``exchange``,
    ``process`` and ``metric`` columns (and ``via`` after pass-through
    reattribution). Where the table came from is irrelevant -- see the
    module docstring for the pipeline this sits in.

    All transformation methods return a *new* Report, so intermediate views
    can be shared and chained::

        staged = Report.elementary_flows(model.structure).with_group("stage", stages, on="process")
        gwp = staged.characterise(GWP, name="GWP")
        gwp.total()                 # scalar sympy expression
        gwp.by("stage", "exchange") # MultiIndex Series of sympy expressions

    Results are always symbolic expressions over structural symbols; evaluate
    them separately with :func:`evaluate_views` / :func:`lambdify_views` against
    a model, or manipulate them directly with sympy. The wrapped table is
    available as ``.table`` if required for additional processing.

    :param table: Tidy DataFrame with a ``value`` column of sympy expressions.
    :param structure: The :class:`~flowprog.ModelStructure` the table's
        symbols refer to.
    :param characterisation: Name of the characterisation applied so far, or
        None. Set by :meth:`characterise`.

    """

    def __init__(
        self,
        table: pd.DataFrame,
        structure=None,
        characterisation: Optional[str] = None,
    ):
        if "value" not in table.columns:
            raise ValueError("Report table must have a 'value' column")
        self.table = table
        self.structure = structure
        self.characterisation = characterisation

    def __repr__(self):
        return (
            f"Report({len(self.table)} rows, columns={list(self.table.columns)}, "
            f"characterisation={self.characterisation!r})"
        )

    @classmethod
    def elementary_flows(cls, structure, table: Optional[pd.DataFrame] = None) -> "Report":
        """Report over the structural elementary-flow table.

        :param structure: A :class:`~flowprog.ModelStructure`.
        :param table: Optional replacement flow table of the same shape, e.g.
            :meth:`flowprog.allocation.PassThrough.elementary_flows` --
            reporting is identical whether or not burdens have been
            reattributed. Defaults to
            ``structure.elementary_flow_table()``.
        """
        if table is None:
            table = structure.elementary_flow_table()
        return cls(table, structure=structure)

    @classmethod
    def production(cls, structure, object_id: str) -> "Report":
        """Report over an object's production (`Y[j]*S[i,j]` per producer).

        The technosphere analogue of :meth:`elementary_flows`: one row per
        producing process, with structural values, so the same grouping/
        filtering/evaluation machinery applies (e.g. utility requirements
        by stage).

        :param structure: A :class:`~flowprog.ModelStructure`.
        """
        return cls._technosphere(structure, "ProcessOutput", object_id)

    @classmethod
    def consumption(cls, structure, object_id: str) -> "Report":
        """Report over an object's consumption (`X[j]*U[i,j]` per consumer).

        See :meth:`production` -- the same, mirrored for the consumption side.
        """
        return cls._technosphere(structure, "ProcessInput", object_id)

    @classmethod
    def _technosphere(cls, structure, flow_role: str, object_id: str) -> "Report":
        process_ids = (
            structure.producers_of(object_id)
            if flow_role == "ProcessOutput"
            else structure.consumers_of(object_id)
        )
        values = [
            structure.expr(flow_role, process_id=pid, object_id=object_id)
            for pid in process_ids
        ]
        table = pd.DataFrame(
            {"object": object_id, "process": process_ids, "value": values}
        )
        return cls(table, structure=structure)

    def _derive(self, table: pd.DataFrame, **overrides) -> "Report":
        kwargs = dict(
            structure=self.structure,
            characterisation=self.characterisation,
        )
        kwargs.update(overrides)
        return Report(table, **kwargs)

    def _id_universe(self, column: str) -> Optional[set]:
        """The full universe of valid ids for a table column, if known."""
        if self.structure is None:
            return None
        if column == "process":
            return {p.id for p in self.structure.processes}
        if column == "exchange":
            return {e.id for e in self.structure.elementary_exchanges}
        if column == "object":
            return {o.id for o in self.structure.objects}
        return None

    def filter(self, **criteria) -> "Report":
        """Restrict to rows whose column values are in the given sets.

        :param criteria: ``column_name=allowed`` pairs; `allowed` may be a
            single value or an iterable of values. Multiple criteria are
            combined with AND.

        **Example**::

            rep.filter(process={"Dirty", "Clean"})
            rep.filter(exchange="CO2")
        """
        mask = pd.Series(True, index=self.table.index)
        for column, allowed in criteria.items():
            if column not in self.table.columns:
                raise ValueError(
                    f"filter: no column {column!r} in table "
                    f"(columns: {list(self.table.columns)})"
                )
            if isinstance(allowed, str) or not isinstance(allowed, Iterable):
                allowed = {allowed}
            mask &= self.table[column].isin(set(allowed))
        return self._derive(self.table[mask].reset_index(drop=True))

    def with_group(
        self,
        name: str,
        groups: dict[str, Iterable[str]],
        on: str,
        other_label: str = "other",
    ) -> "Report":
        """Add a grouping label column derived from an existing id column.

        The grouping is validated with :meth:`Grouping.build`: groups must be
        disjoint, and ids not covered by any group fall into `other_label`
        (with a warning). When the id universe for `on` is known from the
        attached structure (the "process"/"exchange"/"object" columns),
        group ids are also checked against it.

        :param name: Name of the new label column (used in :meth:`by`).
        :param groups: ``{label: ids}`` -- a partition of the `on` column's ids.
        :param on: Existing column the ids refer to (e.g. "process" or
            "exchange").
        :param other_label: Label for ids not covered by any group.
        """
        if on not in self.table.columns:
            raise ValueError(
                f"with_group: no column {on!r} in table to group on "
                f"(columns: {list(self.table.columns)})"
            )
        universe = self._id_universe(on)
        if universe is None:
            # No universe known: accept any group ids (they may legitimately
            # not appear in this table), validate disjointness only.
            universe = set(self.table[on].dropna()) | {
                id_ for ids in groups.values() for id_ in ids
            }
        mapping = Grouping.build(name, groups, universe, other_label)
        labels = [mapping.get(v, other_label) for v in self.table[on]]
        return self._derive(self.table.assign(**{name: labels}))

    def characterise(self, factors, name: Optional[str] = None, on: str = "exchange") -> "Report":
        """Scale each row's value by a per-id factor (e.g. GWP per exchange).

        After characterisation all rows are commensurable (they share the
        characterisation's unit), so :meth:`total`/:meth:`by` will sum across
        different exchanges. Pass a scalar ``1`` to state explicitly that the
        raw values are directly addable (a characterisation vector of ones).

        :param factors: ``{id: factor}`` keyed on the `on` column -- ids
            absent from `factors` get factor 0 (contribute nothing) -- or a
            single scalar applied to every row.
        :param name: Optional name recorded as ``.characterisation`` (defaults
            to ``"characterised"``).
        :param on: Column the factor keys refer to (default "exchange").
        """
        if isinstance(factors, Mapping):
            if on not in self.table.columns:
                raise ValueError(
                    f"characterise: no column {on!r} in table to key factors on "
                    f"(columns: {list(self.table.columns)})"
                )
            values = [
                sy.S(factors.get(k, 0)) * v
                for k, v in zip(self.table[on], self.table["value"])
            ]
        else:
            factor = sy.S(factors)
            values = [factor * v for v in self.table["value"]]
        return self._derive(
            self.table.assign(value=values),
            characterisation=name if name is not None else "characterised",
        )

    def total(self) -> sy.Expr:
        """Sum all values into a single symbolic expression.

        :raises ValueError: If this would sum distinct uncharacterised
            exchange types -- see :meth:`characterise`.
        """
        self._check_commensurable(())
        return sy.Add(*self.table["value"])

    def by(self, *keys: str) -> pd.Series:
        """Group-by-and-sum values, returning a Series of symbolic expressions.

        A plain pandas group-by over the named columns (id columns like
        "process"/"exchange", or label columns added by :meth:`with_group`),
        summing the sympy values in each group. Multiple keys give a
        MultiIndex Series.

        :param keys: Column names to group by.
        :raises ValueError: If a group would sum distinct uncharacterised
            exchange types -- see :meth:`characterise`.
        """
        if not keys:
            raise ValueError("by() needs at least one column name; use total() for a scalar")
        for key in keys:
            if key not in self.table.columns:
                raise ValueError(
                    f"by: no column {key!r} in table "
                    f"(columns: {list(self.table.columns)})"
                )
        self._check_commensurable(keys)
        grouped = self.table.groupby(list(keys), dropna=False)["value"].agg(
            lambda values: sy.Add(*values)
        )
        if len(keys) == 1:
            # groupby with a length-1 list still yields a flat index
            grouped = grouped.rename_axis(keys[0])
        return grouped

    def _check_commensurable(self, keys):
        """Refuse to sum distinct exchange types without a characterisation.

        Different exchange types in general represent different quantities in
        different units; adding them needs a characterisation vector (which
        may deliberately be all ones -- ``characterise(1)``).
        """
        if self.characterisation is not None or "exchange" not in self.table.columns:
            return
        if "exchange" in keys:
            return
        if keys:
            n_exchanges = self.table.groupby(list(keys), dropna=False)["exchange"].nunique()
            mixed = n_exchanges[n_exchanges > 1]
            if mixed.empty:
                return
            where = f"groups {list(mixed.index)}"
        else:
            if self.table["exchange"].nunique() <= 1:
                return
            where = "the total"
        raise ValueError(
            f"Summing {where} would add up different exchange types "
            f"({sorted(self.table['exchange'].unique())}) without a "
            "characterisation; distinct exchanges are in general not "
            "commensurable. Apply .characterise(factors) first, use "
            ".characterise(1) to state that raw values are directly "
            "addable, or include 'exchange' in by()."
        )


def evaluate_views(model, views, values: Optional[dict] = None):
    """Resolve symbolic views against a model, preserving their structure.

    Maps ``model.eval(expr, values)`` over the leaves of `views`: a bare
    expression, a Series (e.g. from :meth:`Report.by`), a DataFrame with a
    ``value`` column (e.g. ``report.table``), or a dict of any of these
    (nested dicts allowed). With ``values=None`` the result is fully-resolved
    symbolic; with numeric `values` it is numeric. For repeated evaluation
    with different values (e.g. Monte Carlo), compile once with
    :func:`lambdify_views` instead.

    :param model: The evaluator that resolves the views' symbols -- anything
        with an ``eval(expr, values)`` method. Usually an evaluable model
        (e.g. SympyModel); for the numpyro backend, pass a forward-run
        ``NumpyroState`` (with ``values=None`` -- parameters are bound when
        the state is produced).
    :param views: Expression, Series, DataFrame, or (nested) dict of these.
    :param values: Optional parameter values to substitute.
    """
    if isinstance(views, Mapping):
        return {k: evaluate_views(model, v, values) for k, v in views.items()}
    if isinstance(views, pd.DataFrame):
        return views.assign(value=[model.eval(sy.S(v), values) for v in views["value"]])
    if isinstance(views, pd.Series):
        return pd.Series(
            [model.eval(sy.S(v), values) for v in views],
            index=views.index,
            name=views.name,
        )
    return model.eval(sy.S(views), values)


def lambdify_views(model, views, modules=None):
    """Compile views into one fast numeric function of the model parameters.

    All expressions are compiled in a single ``model.lambdify()`` pass (so
    shared intermediate expressions are evaluated once), and the returned
    function reassembles results into the same structure as `views`: scalars
    stay scalars, Series keep their index. Compile once, evaluate many times
    -- e.g. in a Monte Carlo loop.

    :param model: The evaluable model that resolves the views' structural
        symbols and provides ``lambdify()`` (e.g. SympyModel).
    :param views: Expression, Series, or a dict of these.
    :param modules: Passed to ``model.lambdify()`` (use ``"math"`` for large
        models with nested Piecewise expressions).
    :return: Function ``values -> results`` mirroring the shape of `views`.
    """
    single = not isinstance(views, Mapping)
    if single:
        views = {None: views}

    flat: dict = {}
    layout: dict = {}
    for name, view in views.items():
        if isinstance(view, pd.Series):
            for position, expr in enumerate(view.values):
                flat[(name, position)] = sy.S(expr)
            layout[name] = view.index
        else:
            flat[(name, None)] = sy.S(view)
            layout[name] = None

    func = model.lambdify(expressions=flat, modules=modules)

    def evaluate(values: dict):
        results = func(values)
        out = {}
        for name, index in layout.items():
            if index is None:
                out[name] = results[(name, None)]
            else:
                out[name] = pd.Series(
                    [results[(name, position)] for position in range(len(index))],
                    index=index,
                )
        return out[None] if single else out

    return evaluate
