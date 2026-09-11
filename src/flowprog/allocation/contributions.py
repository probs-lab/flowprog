"""Contribution analysis: breaking one burden into named parts.

An `AllocatedSystem` says how much burden a unit of each object carries; its
``contributions()`` method says where that burden came from, by choosing a set
of processes to *expand*:

- every expanded process is reported with its own **direct** burdens;
- every input those processes take that is not supplied by another expanded
  process is collapsed into its **upstream** (cradle-to-gate) burden;

Where an object is supplied partly by expanded processes and partly not,
emissions for that object will be split between direct and upstream
contributions. The two kinds of contribution together sum up to the original
burden.

Choosing what to expand
-----------------------

A `ProcessSet` is a plain set of process ids, so it can always be written out by
hand. Alternatively there are some functions to help construct the set based on
the model structure.

This includes the set of processes that produce the object of interest::

    ProcessSet.producers_of(structure, "VinylChloride")     # first tier
    ProcessSet.tiers(structure, 2, object="VinylChloride")  # and their suppliers

To say instead where the breakdown should *stop*::

    ProcessSet.upstream_of(structure, object="VinylChloride",
                           stopping_at_objects=["Electricity", "NaturalGas"])

Sets combine with ``|`` and ``-``, so you can also construct more complicated
sets of processes to expand. For example, to further expand the production of
ethylene, while keeping everything else at the first tier::

    expand = (ProcessSet.producers_of(structure, "VinylChloride") |
              ProcessSet.producers_of(structure, "Ethylene"))

Because a `ProcessSet` is built from the declared model structure rather than
from one set of parameter values, the same set can be used for multiple
allocated systems; processes in the set that don't happen to contribute in a
given scenario are harmless.

What is not attributed
----------------------

A contribution breaks down burden that reaches a *product*. Two kinds of burden
do not appear: burden retained by a process with no outputs to allocate it to
(`AllocatedSystem.sinks`), and burden accumulated in a process's stock
(`AllocatedSystem.stock_burden`).

"""

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
import pandas as pd

from ..reporting import Grouping

# Columns identifying a row of `ContributionResult.table`, before the one
# column per burden slice.
_KEY_COLUMNS = ["kind", "process", "object"]


def _inputs_to(structure, process_id: str) -> list:
    return list(structure.processes[structure.lookup_process(process_id)].consumes)


@dataclass(frozen=True)
class ProcessSet:
    """A set of process ids, with constructors that walk a model structure.

    Used to say which processes a contribution analysis should expand; see the
    module docstring.

    :param processes: The process ids.
    """

    processes: frozenset = frozenset()

    def __post_init__(self):
        object.__setattr__(self, "processes", frozenset(self.processes))

    def __or__(self, other) -> "ProcessSet":
        """Union with another set, a process id, or an iterable of them."""
        return ProcessSet(self.processes | ProcessSet.of(other).processes)

    def __sub__(self, other) -> "ProcessSet":
        """Difference from another set, a process id, or an iterable of them."""
        return ProcessSet(self.processes - ProcessSet.of(other).processes)

    def __len__(self) -> int:
        return len(self.processes)

    def __contains__(self, process_id) -> bool:
        return process_id in self.processes

    def __iter__(self):
        return iter(self.processes)

    @classmethod
    def of(cls, processes) -> "ProcessSet":
        """Exactly the given processes.

        :param processes: A `ProcessSet`, a single process id, or an iterable
            of process ids.
        """
        if isinstance(processes, ProcessSet):
            return processes
        if isinstance(processes, str):
            processes = [processes]
        return cls(processes)

    @classmethod
    def producers_of(cls, structure, objects) -> "ProcessSet":
        """The processes producing the given objects.

        Expanding these and breaking down one of those objects gives the first
        tier: the producers' own emissions, plus everything they take in.

        :param structure: A `ModelStructure`.
        :param objects: A single object id, or an iterable of them.
        """
        if isinstance(objects, str):
            objects = [objects]
        return cls(
            {
                process_id
                for object_id in objects
                for process_id in structure.producers_of(object_id)
            }
        )

    @classmethod
    def tiers(
        cls,
        structure,
        n: int,
        *,
        object: Optional[str] = None,
        process: Optional[str] = None,
    ) -> "ProcessSet":
        """`n` tiers of processes upstream of a starting point, breadth-first.

        Tier 1 is the starting object's producers, or the starting process
        itself; each further tier adds the producers of everything the previous
        tier consumes. A process already included is not revisited, so a loop
        in the supply network does not stop the expansion.

        :param structure: A `ModelStructure`.
        :param n: Number of tiers, at least 1.
        :param object: Object to start from.
        :param process: Process to start from. Give exactly one of `object`
            and `process`.
        """
        if (object is None) == (process is None):
            raise ValueError("give exactly one of object= or process=")
        if n < 1:
            raise ValueError(f"n must be at least 1, got {n}")
        included = (
            {process} if process is not None else set(structure.producers_of(object))
        )
        tier = set(included)
        for _ in range(n - 1):
            nxt = {
                producer
                for process_id in tier
                for object_id in _inputs_to(structure, process_id)
                for producer in structure.producers_of(object_id)
            }
            tier = nxt - included
            if not tier:
                break
            included |= tier
        return cls(included)

    @classmethod
    def upstream_of(
        cls,
        structure,
        *,
        object: Optional[str] = None,
        process: Optional[str] = None,
        stopping_at_objects=(),
        stopping_at_processes=(),
    ) -> "ProcessSet":
        """Everything upstream of a starting point, except beyond the stop
        points.

        The inverse of naming the processes directly: say where the breakdown
        should stop, and this returns the processes that reaches. With no stop
        points it reaches the whole supply chain, which attributes the burden
        to the processes that physically emitted it.

        :param structure: A `ModelStructure`.
        :param object: Object to start from.
        :param process: Process to start from. Give exactly one of `object`
            and `process`.
        :param stopping_at_objects: Object ids whose production is not opened
            up: their suppliers are left out.
        :param stopping_at_processes: Process ids to leave out.
        """
        if (object is None) == (process is None):
            raise ValueError("give exactly one of object= or process=")
        stop_objects = set(stopping_at_objects)
        stop_processes = set(stopping_at_processes)
        included: set = set()
        queue = (
            [process] if process is not None else list(structure.producers_of(object))
        )
        while queue:
            process_id = queue.pop()
            if process_id in included or process_id in stop_processes:
                continue
            included.add(process_id)
            for object_id in _inputs_to(structure, process_id):
                if object_id not in stop_objects:
                    queue.extend(structure.producers_of(object_id))
        return cls(included)

    def suppliers(self, structure) -> frozenset:
        """The processes outside this set that supply something it consumes.

        Expanding this set, these are the processes whose burden is reported as
        cradle-to-gate rather than in detail.

        :param structure: A `ModelStructure`.
        """
        return frozenset(
            producer
            for process_id in self.processes
            for object_id in _inputs_to(structure, process_id)
            for producer in structure.producers_of(object_id)
            if producer not in self.processes
        )

    def external_inputs(self, structure) -> frozenset:
        """The objects this set consumes that it does not wholly supply itself.

        Expanding this set, these are the inputs whose burden is collapsed.
        Includes objects with no producer at all.

        :param structure: A `ModelStructure`.
        """
        external = set()
        for process_id in self.processes:
            for object_id in _inputs_to(structure, process_id):
                producers = set(structure.producers_of(object_id))
                if not producers or producers - self.processes:
                    external.add(object_id)
        return frozenset(external)


def contributions_of(
    system,
    *,
    object: Optional[str] = None,
    process: Optional[str] = None,
    amount: float = 1.0,
    expand,
) -> "ContributionResult":
    """Break one of an `AllocatedSystem`'s burdens down.

    Reached as ``AllocatedSystem.contributions()``, which documents the
    arguments.
    """
    if object is None and process is None:
        raise ValueError("give object=, process=, or both, to break down")

    expand = ProcessSet.of(expand)
    expanded = (
        np.array([p in expand.processes for p in system.processes]) & system._in_scope
    )
    share = system._share_per_unit()
    # The burden a unit of each object brings in from un-expanded processes:
    # the share of it they supply, times what they hand on.
    collapsed_intensity = (share * ~expanded) @ system._out_burden.T
    collapses = (share * ~expanded).any(axis=1)
    unsupplied = system._total_borne == 0

    rows: list[tuple] = []
    quantities: list[tuple] = []
    seed = np.zeros(len(system.processes))

    if process is None:
        i = system.object_index(object)
        if unsupplied[i]:
            raise ValueError(
                f"Object {object} has no modelled supply at this operating "
                "point, so it carries no burden to break down"
            )
        seed = amount * (share * expanded)[i, :]
        if collapses[i]:
            # Part of the object's own supply comes from an un-expanded
            # process, so it is reported like any other collapsed input --
            # with no consuming process, since it is itself what was asked for.
            rows.append(("upstream", None, object, amount * collapsed_intensity[i, :]))
            quantities.append((None, object, amount))
        total = amount * system.object_intensities.values[i, :]
    else:
        j = system.process_index(process)
        if not expanded[j]:
            raise ValueError(
                f"Process {process} is not being expanded, so its burden would "
                "not be broken down at all"
            )
        if object is None:
            if system._reference_activity[j] == 0:
                raise ValueError(
                    f"Process {process} has no activity at this operating "
                    "point, so there is nothing to report a burden per unit of"
                )
            scale = amount / system._reference_activity[j]
        else:
            i = system.object_index(object)
            if system._borne[i, j] == 0:
                raise ValueError(f"Process {process} does not produce {object}")
            scale = amount * system._weights[i, j] / system._borne[i, j]
        seed[j] = scale
        total = scale * system._out_burden[:, j]

    # How much of each expanded process's activity is called for. Operating
    # process j calls for `carried[k, j]` of object k, and the share of that
    # supplied by expanded processes calls for their activity in turn -- so the
    # levels solve `levels = seed + levels @ calls_for`.
    interior = np.nonzero(expanded)[0]
    calls_for = system._carried.T @ (share * expanded)
    levels = np.zeros(len(system.processes))
    if len(interior):
        levels[interior] = np.linalg.solve(
            np.eye(len(interior)) - calls_for[np.ix_(interior, interior)].T,
            seed[interior],
        )

    for j in np.nonzero(levels)[0]:
        emitted = levels[j] * system._direct[:, j]
        rows.append(("direct", system.processes[j], None, emitted))
        for k in np.nonzero(system._carried[:, j])[0]:
            if not (collapses[k] or unsupplied[k]):
                continue  # supplied entirely by expanded processes
            quantity = levels[j] * system._carried[k, j]
            kind = "unsupplied" if unsupplied[k] else "upstream"
            rows.append(
                (
                    kind,
                    system.processes[j],
                    system.objects[k],
                    quantity * collapsed_intensity[k, :],
                )
            )
            quantities.append((system.processes[j], system.objects[k], quantity))

    slices = list(system.slices)
    table = pd.DataFrame(
        [(kind, process_id, object_id, *values) for kind, process_id, object_id, values
         in rows],
        columns=_KEY_COLUMNS + slices,
    )
    return ContributionResult(
        table=table,
        quantities=pd.DataFrame(quantities, columns=["process", "object", "quantity"]),
        total=pd.Series(total, index=slices),
        reference=(object, process),
        amount=amount,
        expanded=expand,
        slices=tuple(slices),
        objects=system.objects,
        processes=system.processes,
    )


@dataclass
class ContributionResult:
    """Where a burden came from, with one set of processes expanded.

    The contributions are held in one table so that they can be grouped and
    checked together, with a ``kind`` column saying what each row is:

    ==========  ============================  ===================  ============
    kind        meaning                       ``process``          ``object``
    ==========  ============================  ===================  ============
    direct      emitted by an expanded        the emitting one     none
                process
    upstream    collapsed burden of an input  the consuming one    the input
    unsupplied  an input with no modelled     the consuming one    the input
                supply
    ==========  ============================  ===================  ============

    There is one further column per burden slice, so different exchanges are
    never summed together by accident. `direct` rows have no input object, so
    ``table["object"]`` is missing for them; use :meth:`by` or
    :meth:`with_group` rather than a bare ``groupby("object")``, which would
    drop them.

    An ``unsupplied`` row is an input that nothing in the model produces. It
    carries no burden -- there is none to carry -- and is reported so that the
    gap is visible rather than silently absent; :meth:`unsupplied` lists how
    much of each was taken in.

    :param table: Contribution rows, as described above.
    :param quantities: ``(process, object, quantity)`` -- how much of each
        collapsed input is charged for.
    :param total: The burden being broken down, per slice.
    :param reference: The ``(object, process)`` that was broken down, either
        of which may be None.
    :param amount: How much of the reference.
    :param expanded: The processes that were expanded.
    :param slices: The burden slice names, which are `table`'s value columns.
    :param objects: Every object id in the model, so that a grouping can be
        checked against them.
    :param processes: Every process id in the model, likewise.
    """

    table: pd.DataFrame
    quantities: pd.DataFrame
    total: pd.Series
    reference: tuple
    amount: float
    expanded: ProcessSet
    slices: tuple
    objects: tuple
    processes: tuple

    def __repr__(self):
        object_id, process_id = self.reference
        what = " from ".join(str(x) for x in (object_id, process_id) if x is not None)
        return (
            f"ContributionResult({self.amount:g} {what}, "
            f"{len(self.expanded)} processes expanded, "
            f"{len(self.table)} contributions)"
        )

    @property
    def direct(self) -> pd.DataFrame:
        """Burden emitted by the expanded processes, by process."""
        rows = self.table[self.table["kind"] == "direct"]
        return rows.set_index("process")[list(self.slices)]

    @property
    def upstream(self) -> pd.DataFrame:
        """Collapsed burden of the inputs, by consuming process and input
        object.

        Sum the process level out for the burden by input object alone::

            result.upstream.groupby("object").sum()
        """
        rows = self.table[self.table["kind"] != "direct"]
        return rows.set_index(["process", "object"])[list(self.slices)]

    def by(self, *keys: str) -> pd.DataFrame:
        """Group the contributions and sum them.

        :param keys: Columns to group by -- ``"kind"``, ``"process"``,
            ``"object"``, or a label column added by :meth:`with_group`. Rows
            with no value for a key (direct rows have no object) are grouped
            under their ``kind``.
        :return: DataFrame of the grouped rows against the burden slices.
        """
        if not keys:
            raise ValueError("by() needs at least one column to group by")
        missing = [key for key in keys if key not in self.table.columns]
        if missing:
            raise ValueError(
                f"by: no column {missing} in the contributions "
                f"(columns: {list(self.table.columns)})"
            )
        table = self.table.copy()
        for key in keys:
            table[key] = table[key].fillna(table["kind"])
        return table.groupby(list(keys), dropna=False)[list(self.slices)].sum()

    def with_group(
        self,
        name: str,
        groups: dict,
        on: str = "object",
        direct_label: str = "direct",
        other_label: str = "other",
    ) -> "ContributionResult":
        """Add a grouping label column, for reporting in named categories.

        Groups are validated as a partition by
        :class:`flowprog.reporting.Grouping`: they must be disjoint, and must
        reference ids that exist in the model. Anything they miss that does
        appear in these contributions is bucketed into `other_label` with a
        warning, so a forgotten id is flagged; naming an id that happens not to
        appear is fine, and simply produces no row.

        **Example**::

            result.with_group("category", {
                "chlorine": ["Chlorine"],
                "feedstocks": ["Ethylene", "Naphtha"],
                "utilities": ["Electricity", "ProcessHeat"],
            }).by("category")

        :param name: Name of the label column to add.
        :param groups: ``{label: ids}`` covering the ids in the `on` column.
        :param on: Column the ids refer to -- ``"object"`` or ``"process"``.
        :param direct_label: Label for rows with no value in `on`, which is
            how direct burdens become a category of their own.
        :param other_label: Label for ids no group covers.
        :return: A new result with the label column added.
        """
        if on not in self.table.columns:
            raise ValueError(
                f"with_group: no column {on} in the contributions "
                f"(columns: {list(self.table.columns)})"
            )
        universe = {"object": set(self.objects), "process": set(self.processes)}.get(on)
        declared = {id_ for ids in groups.values() for id_ in ids}
        if universe is not None and declared - universe:
            raise ValueError(
                f"Grouping {name!r}: group references unknown id(s) "
                f"{sorted(declared - universe)}"
            )
        present = set(self.table[on].dropna())
        mapping = Grouping.build(name, groups, present | declared, other_label)
        labels = [
            direct_label if pd.isna(value) else mapping.get(value, other_label)
            for value in self.table[on]
        ]
        return replace(self, table=self.table.assign(**{name: labels}))

    def unsupplied(self) -> pd.DataFrame:
        """Inputs that nothing in the model produces.

        :return: ``(process, object, quantity)`` for those inputs. They carry
            no burden, so the quantity is all there is to report.
        """
        missing = self.table[self.table["kind"] == "unsupplied"][["process", "object"]]
        return self.quantities.merge(
            missing.drop_duplicates(), on=["process", "object"]
        )

    def check(self, atol: float = 1e-6) -> bool:
        """Check the contributions add up to the burden being broken down.

        :param atol: Absolute tolerance per slice.
        :raises AssertionError: If any slice does not add up.
        :return: True if every slice adds up.
        """
        found = self.table[list(self.slices)].sum()
        bad = {
            name: (found[name], self.total[name])
            for name in self.slices
            if abs(found[name] - self.total[name]) > atol
        }
        if bad:
            raise AssertionError(
                "Contributions do not add up to the burden being broken down "
                f"(found, expected): {bad}"
            )
        return True
