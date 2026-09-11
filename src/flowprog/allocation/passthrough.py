"""Pass-through: symbolic reattribution of burdens to direct consumers."""

import pandas as pd
import sympy as sy


class PassThrough:
    """Reattribute pass-through processes' burdens to their direct consumers.

    Conceptually this is allocation with a limited propagation frontier: the
    processes in `pass_through` are treated as transient rather than reporting
    categories in their own right, and their elementary-exchange burdens are
    pushed forward to the processes consuming their output, in proportion to
    consumption. Every process outside the set keeps its own burden. Since
    burden is moved exactly once, any grouping over the reattributed table still
    partitions the system total. This is similar to GHG Protocol scope-2-style
    attribution of purchased electricity/heat to the processes that use them.

    **Currently implemented:** the closed-form degenerate case, where every
    pass-through process has no technosphere inputs, exactly one output, and
    is the sole supplier of that output (e.g. a generated boundary `Source`
    from `flowprog.boundary_processes`). For such processes the embodied
    intensity of the supplied object is simply ``B[e, s] / S[i, s]`` -- the
    general allocation solve degenerates to it exactly -- so reattribution is
    symbolic, division-free in the unit-S case, and stays on the model's
    lambdify-once fast path (and remains compilable to standalone code).

    **Not yet implemented**:

    - multi-output pass-through processes, which need an allocation `rule`
      to split their burden among co-products;
    - objects with more than one supplying process, which need market-share
      weighting (division by total supply, with zero-supply guards);
    - pass-through processes with technosphere inputs, which need a
      forward-substitution solve over the pass-through set (acyclic case)
      or closed-form small-cycle solves.

    Consumption is keyed to input activity ``X[j] * U[i, j]``. Totals per
    exchange are preserved exactly iff each pass-through object's market
    balances -- check `residuals()`.

    The allocation depends only on the model *structure*: every output value is
    a purely structural symbolic expression (``X[j]``, ``U[i,j]``, ``B[e,s]``,
    ``S[i,s]``), resolved later when the model is evaluated.

    :param structure: A `ModelStructure`.
    :param pass_through: Iterable of process ids to reattribute.
    :param rule: Reserved for multi-output pass-through processes; must be
        None for now.

    Reporting composes with this by table substitution: `elementary_flows()`
    has the same shape as ``structure.elementary_flow_table()``, so pass it
    to `flowprog.reporting.Report.elementary_flows` and report exactly as for
    an unmodified model.

    **Example**::

        from flowprog.reporting import Report

        pt = PassThrough(model.structure, ["SourceOfElectricity", "SourceOfProcessHeat"])
        rep = Report.elementary_flows(model.structure, pt.elementary_flows())
        rep.with_group("stage", stages, on="process").by("stage", "exchange")

    """

    def __init__(self, structure, pass_through, rule=None):
        if rule is not None:
            raise NotImplementedError(
                "Allocation rules for multi-output pass-through processes are "
                "not yet implemented; `rule` must be None."
            )
        self.structure = structure
        self.pass_through = list(dict.fromkeys(pass_through))
        self._plan = self._validate()

    def _validate(self):
        """Check every pass-through process is in the supported closed-form
        case, returning [(process_id, object_id, consumer_ids)]."""
        structure = self.structure
        plan = []
        for pid in self.pass_through:
            j = structure.lookup_process(pid)  # raises ValueError if unknown
            process = structure.processes[j]

            if len(process.produces) != 1:
                raise NotImplementedError(
                    f"Pass-through process {pid!r} produces "
                    f"{len(process.produces)} objects; only single-output "
                    "pass-through processes are supported (multi-output needs "
                    "an allocation rule, not yet implemented)."
                )
            if process.consumes:
                raise NotImplementedError(
                    f"Pass-through process {pid!r} has technosphere inputs "
                    f"({process.consumes}); pass-through processes with "
                    "inputs need a forward-substitution solve over the "
                    "pass-through set, not yet implemented."
                )

            object_id = process.produces[0]
            other_suppliers = [
                q for q in structure.producers_of(object_id) if q != pid
            ]
            if other_suppliers:
                raise NotImplementedError(
                    f"Object {object_id!r} (supplied by pass-through process "
                    f"{pid!r}) is also supplied by {other_suppliers}; "
                    "objects with multiple suppliers need market-share "
                    "weighting, not yet implemented."
                )

            consumers = structure.consumers_of(object_id)
            plan.append((pid, object_id, consumers))
        return plan

    def _intensities(self, pid, object_id):
        """{exchange_id: burden per unit of `object_id` supplied} for one
        pass-through process (closed form: ``B[e,s] / S[i,s]``, symbolic).

        Iterates the process's declared exchanges (`Process.exchanges`); a
        declared exchange with no recipe value simply resolves to zero
        burden per unit at evaluation time.
        """
        structure = self.structure
        s = structure.lookup_process(pid)
        i = structure.lookup_object(object_id)
        return {
            exchange_id: structure.B[structure.lookup_exchange(exchange_id), s]
            / structure.S[i, s]
            for exchange_id in structure.processes[s].exchanges
        }

    def elementary_flows(self) -> pd.DataFrame:
        """Reattributed elementary-flow table (structural symbolic values).

        Same shape as ``structure.elementary_flow_table()`` plus a ``via``
        column, except that each pass-through process's rows are replaced by
        one row per (exchange, consumer): ``X[j]*U[i,j] * B[e,s]/S[i,s]``,
        with ``via`` naming the pass-through process the burden was received
        through (NA on processes' own direct rows). Every value is a purely
        structural symbolic expression, ready for
        ``lambdify(expressions=...)``, ``eval()``, or a forward-run state.
        """
        structure = self.structure
        pass_through_ids = {pid for pid, _, _ in self._plan}

        table = structure.elementary_flow_table()
        table = table[~table["process"].isin(pass_through_ids)].copy()
        table["via"] = pd.NA

        rows = []
        for pid, object_id, consumers in self._plan:
            i = structure.lookup_object(object_id)
            intensities = self._intensities(pid, object_id)
            for exchange_id, intensity in intensities.items():
                e = structure.lookup_exchange(exchange_id)
                metric = structure.elementary_exchanges[e].metric
                for q in consumers:
                    jq = structure.lookup_process(q)
                    value = structure.X[jq] * structure.U[i, jq] * intensity
                    rows.append((exchange_id, q, metric, value, pid))

        received = pd.DataFrame(
            rows, columns=["exchange", "process", "metric", "value", "via"]
        )
        return pd.concat([table, received], ignore_index=True)

    def residuals(self) -> pd.DataFrame:
        """Burden booked on each pass-through process minus burden
        redistributed to its consumers, per exchange (raw symbolic values).

        Equals ``B[e,s]/S[i,s] *`` (the object's market balance), so every
        value evaluates to zero iff supply of the pass-through object exactly
        matches consumption. Nonzero residual burden is *dropped* from
        `elementary_flows()` -- evaluate this table in tests to confirm the
        reattribution is exact.
        """
        structure = self.structure
        rows = []
        for pid, object_id, consumers in self._plan:
            js = structure.lookup_process(pid)
            i = structure.lookup_object(object_id)
            supplied = structure.Y[js] * structure.S[i, js]
            consumed = sum(
                (
                    structure.X[jq] * structure.U[i, jq]
                    for jq in (structure.lookup_process(q) for q in consumers)
                ),
                sy.S.Zero,
            )
            for exchange_id, intensity in self._intensities(pid, object_id).items():
                rows.append((exchange_id, pid, intensity * (supplied - consumed)))
        return pd.DataFrame(rows, columns=["exchange", "process", "value"])
