"""Allocation: attributional average burdens (object intensities mu, process
intensities beta) at a numeric operating point, plus symbolic pass-through
reattribution of burdens to direct consumers.

Two tools, one mathematical family:

- `Allocation`: full-chain mu/beta, solved numerically at a parameter point.
  The model is evaluated, floats are extracted, and (I - M) mu = c is solved
  by dense linear algebra -- the system is tiny (one row/column per object),
  so this is fast even repeated across many parameter samples (the numeric
  extraction is compiled once via the model's lambdify path and cached on the
  model instance). See the implementation plan section 6 for the mathematics.

- `PassThrough`: allocation with a *limited propagation frontier*. A
  designated set of "pass-through" processes have their burdens (per unit of
  their output) pushed to their direct consumers in proportion to
  consumption; every other process retains its own burden. Because burden is
  moved, never copied, any grouping over the reattributed (exchange x
  process) table still partitions the system total -- unlike grouping
  full-chain beta*Y, which double counts embodied burdens across groups.
  Implemented symbolically, so results stay on the model's
  lambdify-once/evaluate-many fast path and can be compiled to standalone
  code.
"""

import logging
import weakref
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import sympy as sy

_log = logging.getLogger(__name__)

# Per-model cache of the compiled numeric-extraction function (see
# _numeric_arrays()), keyed by model instance rather than as an attribute on
# the model itself -- this module only reads the model's public interface.
_extraction_cache: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


@dataclass(frozen=True)
class Scope:
    """Selects which processes participate in an allocation solve.

    :param excluded_processes: Process ids excluded entirely from the solve
        (e.g. a designated end-of-life set, giving cradle-to-gate mu).
    :param waste_objects: Object ids whose *input* burden into any in-scope
        process is governed by `waste_input_burden`.
    :param waste_input_burden: "cutoff" (default -- matches ecoinvent cut-off:
        waste objects carry zero burden as an input, breaking recycling
        loops) or "propagate" (waste objects carry their own solved
        intensity as an input, like any other object).
    """

    excluded_processes: frozenset = frozenset()
    waste_objects: frozenset = frozenset()
    waste_input_burden: str = "cutoff"

    def __post_init__(self):
        if self.waste_input_burden not in ("cutoff", "propagate"):
            raise ValueError(
                "waste_input_burden must be 'cutoff' or 'propagate', got "
                f"{self.waste_input_burden!r}"
            )

    def process_included(self, process_id: str) -> bool:
        return process_id not in self.excluded_processes


class MassAllocation:
    """Allocation weights proportional to mass (S_ij) flow."""

    def raw_weight(self, object_id, process_id, S_ij):
        return S_ij


@dataclass
class PropertyAllocation:
    """Allocation weights proportional to S_ij times a per-object property
    (e.g. energy content, price).

    :param properties: {object_id: property value}. A multi-output process
        with an output missing from `properties` is an uncovered process.
    """

    properties: dict

    def raw_weight(self, object_id, process_id, S_ij):
        if object_id not in self.properties:
            return None
        return S_ij * self.properties[object_id]


@dataclass
class ManualAllocation:
    """Explicit allocation weights.

    :param weights: {process_id: {object_id: weight}}. A multi-output process
        absent from `weights` (or missing one of its outputs) is uncovered.
    """

    weights: dict

    def raw_weight(self, object_id, process_id, S_ij):
        proc_weights = self.weights.get(process_id)
        if proc_weights is None or object_id not in proc_weights:
            return None
        return proc_weights[object_id]


@dataclass
class AllocationResult:
    """Result of an Allocation solve.

    :param object_intensities: "mu" -- DataFrame: objects x slices
    :param process_intensities: "beta" -- DataFrame: processes x slices
    :param supply_shares: "sigma" -- long-form DataFrame (object, process, sigma)
    :param meta: dict with keys "rule", "scope", "cutoffs" (list of
        (process_id, object_id) pairs with a deliberate zero-weight cutoff on
        a nonzero flow), "zero_supply_objects", "conservation_residuals"
    """

    object_intensities: pd.DataFrame
    process_intensities: pd.DataFrame
    supply_shares: pd.DataFrame
    meta: dict

    def check_conservation(self, atol: float = 1e-6) -> bool:
        """Check the 100% rule for every slice.

        Within scope: Sum_i mu_i * boundary_output_i + sink/stock terms ==
        Sum_j(in scope) b_j * Y_j (raw, uncharacterised-consistent per slice).

        :raises AssertionError: If conservation fails for any slice beyond `atol`.
        :return: True if conservation holds for all slices.
        """
        residuals = self.meta["conservation_residuals"]
        bad = {k: v for k, v in residuals.items() if not np.isnan(v) and abs(v) > atol}
        if bad:
            raise AssertionError(f"Conservation failed for slices: {bad}")
        return True


class Allocation:
    """Solve attributional average burdens (mu, beta) at a numeric operating point.

    :param model: A built, evaluable model (e.g. SympyModel)
    :param values: Numeric parameter values at which to evaluate (the
        operating point theta)
    :param rule: An allocation rule (MassAllocation/PropertyAllocation/
        ManualAllocation) for splitting multi-output processes. No default --
        the choice must be explicit.
    :param scope: Optional Scope selecting participating processes and
        waste-object cut-off behaviour. Default: all processes in scope, no
        waste objects.
    :param characterise: Optional ``{name: {exchange_id: factor}}`` extra
        characterised slices, in addition to the default per-exchange
        breakdown (mu/beta always carry the full per-exchange breakdown).

    Access results via `.result` (an AllocationResult).
    """

    def __init__(
        self,
        model,
        values: dict,
        rule,
        scope: Optional[Scope] = None,
        characterise: Optional[dict] = None,
    ):
        self.model = model
        self.values = values
        self.rule = rule
        self.scope = scope or Scope()
        self.characterise = dict(characterise or {})

        self.result = self._solve()

    def _solve(self) -> AllocationResult:
        model = self.model
        processes = model.processes
        objects = model.objects
        exchanges = model.structure.elementary_exchanges
        M, N = len(processes), len(objects)

        X, Y, S, U, B = _numeric_arrays(model, self.values)

        in_scope = np.array([self.scope.process_included(p.id) for p in processes])
        waste_idx = {model.structure.lookup_object(o) for o in self.scope.waste_objects}

        # Effective U: cut off waste-object *input* burden if configured.
        U_eff = U.copy()
        if self.scope.waste_input_burden == "cutoff":
            for i in waste_idx:
                U_eff[i, :] = 0.0

        T = np.array(
            [sum(Y[j] * S[i, j] for j in range(M) if in_scope[j]) for i in range(N)]
        )

        weights, cutoffs = _compute_weights(self.rule, processes, objects, model, S, in_scope)

        zero_supply = [i for i in range(N) if T[i] == 0]
        for i in zero_supply:
            _log.warning(
                "Allocation: object %r has zero total in-scope supply; its "
                "intensity will be NaN.",
                objects[i].id,
            )

        # M[i, k] = sum_{j produces i, in scope} (Y_j * w_ij / T_i) * U_eff[k, j]
        # (tag-independent: solved once for all slices as multiple RHS).
        Mmat = np.zeros((N, N))
        for i in range(N):
            if T[i] == 0:
                continue
            for j in range(M):
                if not in_scope[j] or weights[i, j] == 0:
                    continue
                Mmat[i, :] += (Y[j] * weights[i, j] / T[i]) * U_eff[:, j]

        slice_names = [exc.id for exc in exchanges] + list(self.characterise.keys())
        q_vectors = {}
        for e, exc in enumerate(exchanges):
            q = np.zeros(len(exchanges))
            q[e] = 1.0
            q_vectors[exc.id] = q
        for name, factors in self.characterise.items():
            q_vectors[name] = np.array([factors.get(exc.id, 0) for exc in exchanges])

        b = {name: q_vectors[name] @ B for name in slice_names}

        C = np.zeros((N, len(slice_names)))
        for col, name in enumerate(slice_names):
            bt = b[name]
            for i in range(N):
                if T[i] == 0:
                    C[i, col] = np.nan
                    continue
                total = 0.0
                for j in range(M):
                    if not in_scope[j] or weights[i, j] == 0:
                        continue
                    total += Y[j] * weights[i, j] / T[i] * bt[j]
                C[i, col] = total

        # Zero-supply objects are excluded from the dense solve entirely
        # (not just given a NaN row) -- a NaN RHS entry inside a full
        # np.linalg.solve can contaminate unrelated unknowns via LAPACK's
        # internal pivoting, which would violate "NaN cannot propagate into
        # any nonzero-weighted result". Excluding the row/column keeps
        # genuinely-unrelated objects unaffected; their own mu is set to NaN
        # directly afterwards.
        mu = np.full((N, len(slice_names)), np.nan)
        solvable = [i for i in range(N) if T[i] != 0]
        if solvable:
            A = np.eye(len(solvable)) - Mmat[np.ix_(solvable, solvable)]
            sub_mu = np.linalg.solve(A, C[solvable, :])
            for row, i in enumerate(solvable):
                mu[i, :] = sub_mu[row, :]

        # beta_j = b_j + sum_i U_eff[i,j] * mu_i (per unit Y_j; plain
        # elementwise arithmetic here, so a genuine dependency on a
        # zero-supply object correctly and safely yields NaN for this process
        # only, without any matrix-solve contamination risk).
        beta = np.full((M, len(slice_names)), np.nan)
        for col, name in enumerate(slice_names):
            bt = b[name]
            for j in range(M):
                if not in_scope[j]:
                    continue
                total = bt[j]
                for i in range(N):
                    if U_eff[i, j] != 0:
                        total += U_eff[i, j] * mu[i, col]
                beta[j, col] = total

        residuals = _check_conservation_residuals(
            processes, X, Y, U_eff, T, in_scope, mu, beta, b, slice_names
        )

        object_intensities = pd.DataFrame(
            mu, index=[o.id for o in objects], columns=slice_names
        )
        process_intensities = pd.DataFrame(
            beta, index=[p.id for p in processes], columns=slice_names
        )

        sigma_rows = [
            (objects[i].id, processes[j].id, Y[j] * S[i, j] / T[i])
            for i in range(N)
            for j in range(M)
            if S[i, j] > 0 and in_scope[j] and T[i] > 0
        ]
        supply_shares = pd.DataFrame(sigma_rows, columns=["object", "process", "sigma"])

        meta = {
            "rule": self.rule,
            "scope": self.scope,
            "cutoffs": cutoffs,
            "zero_supply_objects": [objects[i].id for i in zero_supply],
            "conservation_residuals": residuals,
        }

        return AllocationResult(
            object_intensities=object_intensities,
            process_intensities=process_intensities,
            supply_shares=supply_shares,
            meta=meta,
        )


def _compute_weights(rule, processes, objects, model, S, in_scope):
    """Compute normalised weights w[i,j] and record deliberate zero-weight cutoffs.

    Single-output processes always get weight 1 for their one output,
    regardless of rule. Multi-output processes ask the rule for each output's
    raw (unnormalised) weight; missing data across all such processes is
    collected into one combined error.
    """
    M = len(processes)
    N = len(objects)
    weights = np.zeros((N, M))
    errors = []

    for j, p in enumerate(processes):
        if not in_scope[j]:
            continue
        out_idx = [model.structure.lookup_object(o) for o in p.produces]
        if len(out_idx) == 0:
            continue
        if len(out_idx) == 1:
            weights[out_idx[0], j] = 1.0
            continue

        raws = {}
        missing = []
        for i in out_idx:
            w = rule.raw_weight(objects[i].id, p.id, S[i, j])
            if w is None:
                missing.append(objects[i].id)
            else:
                raws[i] = w
        if missing:
            errors.append((p.id, missing))
            continue

        total = sum(raws.values())
        if total == 0:
            errors.append((p.id, [objects[i].id for i in out_idx]))
            continue
        for i, w in raws.items():
            weights[i, j] = w / total

    if errors:
        details = "; ".join(f"{pid} (missing: {objs})" for pid, objs in errors)
        raise ValueError(
            f"Allocation rule does not cover all multi-output processes: {details}"
        )

    cutoffs = []
    for j, p in enumerate(processes):
        if not in_scope[j]:
            continue
        for obj_id in p.produces:
            i = model.structure.lookup_object(obj_id)
            if S[i, j] > 0 and weights[i, j] == 0:
                cutoffs.append((p.id, obj_id))

    return weights, cutoffs


def _check_conservation_residuals(processes, X, Y, U_eff, T, in_scope, mu, beta, b, slice_names):
    """Per-slice: Sum_i mu_i*boundary_output_i + sink_terms - Sum_j(in scope) b_j*Y_j.

    Should be ~0 for every slice if mu/beta were solved consistently.
    boundary_output_i = T_i - sum_{j in scope} Y_j * U_eff[i,j] (consumption
    deficit relative to the *scoped* subset of processes, using the same
    Y-scaling as the primary mu/beta solve). sink_terms account for burden
    that the per-object mu accounting cannot capture: the entire Y_j*beta_j of
    any in-scope process producing nothing at all (pure removal/export/sink
    processes), plus the has_stock (X_j != Y_j) net-accumulation discrepancy
    for processes that do produce something.
    """
    M = len(processes)
    N = len(T)
    residuals = {}

    for col, name in enumerate(slice_names):
        bt = b[name]

        boundary_output = T - np.array(
            [
                sum(Y[j] * U_eff[i, j] for j in range(M) if in_scope[j])
                for i in range(N)
            ]
        )

        lhs = sum(
            mu[i, col] * boundary_output[i] for i in range(N) if T[i] != 0
        )

        sink_terms = 0.0
        for j, p in enumerate(processes):
            if not in_scope[j]:
                continue
            if len(p.produces) == 0:
                sink_terms += Y[j] * beta[j, col]
            elif X[j] != Y[j]:
                extra = sum(U_eff[i, j] * mu[i, col] for i in range(N) if U_eff[i, j] != 0)
                sink_terms += (X[j] - Y[j]) * extra

        rhs = sum(bt[j] * Y[j] for j in range(M) if in_scope[j])
        residuals[name] = lhs + sink_terms - rhs

    return residuals


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


def _numeric_arrays(model, values):
    """Evaluate the model's Y, X, S, U, B at `values`, returning numpy arrays.

    Compiles a single lambdified extraction function once per model (cached
    in a module-level weak-keyed dict, keyed by model instance) so repeated
    calls -- e.g. across many parameter samples -- are cheap.
    """
    cache = _extraction_cache.get(model)
    if cache is None:
        processes = model.processes
        n_processes = len(processes)
        n_objects = len(model.objects)
        n_exchanges = len(model.structure.elementary_exchanges)

        # Y[j]/X[j] use _get_value() (not eval()): eval() eagerly expands every
        # intermediate in the model via repeated .subs(), which is slow for
        # large models with deeply nested Piecewise expressions. _get_value()
        # leaves recipe *and* intermediate placeholders un-substituted (so
        # Y/X stay separate from the S/U/B read below); lambdify() resolves
        # them all efficiently via its CSE-aware code generation instead.
        #
        # S/U/B use get_recipe() directly rather than eval(model.S[i,j]):
        # eval() unconditionally rebuilds and re-substitutes *every*
        # intermediate in the model on every call (not just ones the
        # expression references), so calling it once per recipe entry pays
        # that full cost hundreds of times over. Recipe values need no
        # intermediate expansion at all -- they're plain data -- so reading
        # them directly and letting lambdify's function-call args pick up
        # any free symbols they contain (e.g. EF_Feedstock_Naphtha) is both
        # correct and far cheaper. sy.S() wraps plain floats as genuine
        # sympy expressions for lambdify().
        exprs = {}
        for j in range(n_processes):
            exprs[("Y", j)] = sy.S(model._get_value(model.Y[j]))
            exprs[("X", j)] = sy.S(model._get_value(model.X[j]))
        for j, p in enumerate(processes):
            recipe = model.get_recipe(p.id)
            for obj_id, value in recipe.get("produces", {}).items():
                i = model.structure.lookup_object(obj_id)
                exprs[("S", i, j)] = sy.S(value)
            for obj_id, value in recipe.get("consumes", {}).items():
                i = model.structure.lookup_object(obj_id)
                exprs[("U", i, j)] = sy.S(value)
            for exchange_id, value in recipe.get("exchanges", {}).items():
                e = model.structure.lookup_exchange(exchange_id)
                exprs[("B", e, j)] = sy.S(value)

        func = model.lambdify(expressions=exprs)
        cache = (func, n_processes, n_objects, n_exchanges)
        _extraction_cache[model] = cache

    func, n_processes, n_objects, n_exchanges = cache
    result = func(values)

    X = np.zeros(n_processes)
    Y = np.zeros(n_processes)
    S = np.zeros((n_objects, n_processes))
    U = np.zeros((n_objects, n_processes))
    B = np.zeros((n_exchanges, n_processes))
    for key, v in result.items():
        v = float(v)
        kind = key[0]
        if kind == "Y":
            Y[key[1]] = v
        elif kind == "X":
            X[key[1]] = v
        elif kind == "S":
            S[key[1], key[2]] = v
        elif kind == "U":
            U[key[1], key[2]] = v
        elif kind == "B":
            B[key[1], key[2]] = v

    return X, Y, S, U, B
