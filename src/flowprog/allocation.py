"""Allocation: attributional average burdens (object intensities mu, process
intensities beta) at a numeric operating point, plus symbolic pass-through
reattribution of burdens to direct consumers.

There are two related calculations:

- `Allocation`: full attributional calculation, solved numerically at a
  parameter point. The model is evaluated, numeric values are extracted, and (I
  - M) mu = c is solved by dense linear algebra.

- `PassThrough`: allocation with a limited propagation frontier. A designated
  set of "pass-through" processes have their burdens pushed to their direct
  consumers in proportion to consumption; every other process retains its own
  burden. Because burden is only moved, any grouping over the reattributed
  (exchange x process) table still partitions the system total. This is a
  special case of the general allocation that can be implemented symbolically,
  so the resulting expressions can be compiled to standalone code.

Stocks
------

A process with `has_stock` can take in more than it puts out, or less (``X !=
Y``). By default, output flows carry burdens based on the input flows scaled to
the output activity, so material flowing out has the same burden intensity as
the material flowing in. The difference in absolute burden is reported per
process as ``meta["stock_burden"]``.

Allocation rules
----------------

`Allocation` splits each process's burden across its outputs using a *rule*. A
rule is asked for the weights of the outputs of a given process, which is
automatically normalised.

The available rules are::

    ByValue()                   proportional to the output quantities
    ByProperty(properties)      ... times a per-object property
    Fixed(shares)               explicit weights
    Excluding(objects, rule)    those objects bear no burden
    Rules(default, by_process)  a default, with exceptions per process

`Rules` is itself a rule, so the pieces nest.  For example::

    Rules(
        default=Excluding({"Air", "Water", "WasteWater"}, ByValue()),
        by_process={
            "CombinedHeatAndPower": Fixed({"Electricity": 0.6, "Heat": 0.4}),
        },
    )

"""

import logging
import weakref
from dataclasses import dataclass, field
from typing import Mapping, Optional, Protocol

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


class Rule(Protocol):
    """Splits one process's burden across its flows."""

    def weights(self, process, flows: Mapping[str, float]) -> Mapping[str, float]:
        """Raw weights, one for each key of `flows`.

        The weights need not sum to one; they are normalised by the caller. A
        weight of 0 means the flow bears none of the burden.

        :param process: The `Process` whose burden is being split.
        :param flows: ``{object id: quantity}`` -- the flows to split the
            burden across.
        :raises ValueError: If the rule has no value for one of the flows.

        """


def _lookup(values, object_id, process, rule):
    if object_id not in values:
        raise ValueError(
            f"{type(rule).__name__} has no value for {object_id!r}, needed to "
            f"split the burden of process {process.id!r}"
        )
    return values[object_id]


@dataclass
class ByValue:
    """Weights proportional to the flows themselves."""

    def weights(self, process, flows):
        return dict(flows)


@dataclass
class ByProperty:
    """Weights proportional to the flows times a per-object property, such as
    energy content, exergy or price.

    :param properties: ``{object id: property value}``, covering every flow of
        every process this rule is asked about.
    """

    properties: dict

    def weights(self, process, flows):
        return {
            object_id: quantity * _lookup(self.properties, object_id, process, self)
            for object_id, quantity in flows.items()
        }


@dataclass
class Fixed:
    """Explicit weights, independent of the flow quantities.

    :param shares: ``{object id: weight}``, covering every flow of every
        process this rule is asked about.
    """

    shares: dict

    def weights(self, process, flows):
        return {
            object_id: _lookup(self.shares, object_id, process, self)
            for object_id in flows
        }


@dataclass
class Excluding:
    """Objects that bear no burden, with another rule splitting the rest.

    Residual outputs -- air, water, wastes -- usually take no share of the
    burden of the process that emits them::

        Excluding({"Air", "Water", "WasteWater"}, ByValue())

    The excluded objects are removed before `rule` is asked, so `rule` does not
    need data for them.

    :param objects: Object ids that always get weight 0.
    :param rule: Splits the burden across the remaining flows.

    """

    objects: frozenset
    rule: Rule

    def __post_init__(self):
        self.objects = frozenset(self.objects)

    def weights(self, process, flows):
        bearing = {
            object_id: quantity
            for object_id, quantity in flows.items()
            if object_id not in self.objects
        }
        weights = self.rule.weights(process, bearing) if bearing else {}
        return {object_id: weights.get(object_id, 0.0) for object_id in flows}

    def validate(self, structure):
        unknown = self.objects - {o.id for o in structure.objects}
        if unknown:
            raise ValueError(
                f"Excluding references unknown objects {sorted(unknown)}"
            )
        _validate(self.rule, structure)


@dataclass
class Rules:
    """A default rule, with exceptions for named processes.

    `Rules` is itself a rule, so it goes anywhere a rule does::

        Rules(
            default=Excluding(RESIDUAL_OUTPUTS, ByValue()),
            by_process={"SteamCracking": ByProperty(energy_content)},
        )

    An entry in `by_process` replaces the default outright for that process:
    each entry states that process's split in full, including which of its
    outputs bear no burden.

    :param default: Rule for processes not named in `by_process`.
    :param by_process: ``{process id: rule}``.
    """

    default: Rule
    by_process: dict = field(default_factory=dict)

    def rule_for(self, process_id) -> Rule:
        """The rule that splits `process_id`'s burden."""
        return self.by_process.get(process_id, self.default)

    def weights(self, process, flows):
        return self.rule_for(process.id).weights(process, flows)

    def validate(self, structure):
        for process_id in self.by_process:
            structure.lookup_process(process_id)  # raises on an unknown id
        _validate(self.default, structure)
        for rule in self.by_process.values():
            _validate(rule, structure)

    def assignments(self, structure) -> pd.DataFrame:
        """Which rule splits which multi-output process's burden.

        Answers "what will this do?" without running a solve.

        :return: DataFrame with columns ``process``, ``outputs``, ``rule``.
        """
        return pd.DataFrame(
            [
                (p.id, len(p.produces), type(self.rule_for(p.id)).__name__)
                for p in structure.processes
                if len(p.produces) > 1
            ],
            columns=["process", "outputs", "rule"],
        )


def _validate(rule, structure):
    """Check a rule's object and process ids against the model, if it can."""
    validate = getattr(rule, "validate", None)
    if validate is not None:
        validate(structure)


@dataclass
class AllocationResult:
    """Result of an Allocation solve.

    :param object_intensities: "mu" -- DataFrame: objects x slices
    :param process_intensities: "beta" -- DataFrame: processes x slices
    :param supply_shares: "sigma" -- long-form DataFrame (object, process, sigma)
    :param meta: dict with keys "rule", "scope", "cutoffs" (list of
        (process_id, object_id) pairs with a deliberate zero-weight cutoff on
        a nonzero flow), "sinks" (processes whose burden is not further allocated),
        "stock_burden" (processes x slices, burden accumulated in or released
        from each process's stock),"zero_supply_objects", "conservation_residuals"
    """

    object_intensities: pd.DataFrame
    process_intensities: pd.DataFrame
    supply_shares: pd.DataFrame
    meta: dict

    def check_conservation(self, atol: float = 1e-6) -> bool:
        """Check the 100% rule for every slice.

        All burdens from the in-scope processes should be accounted for in three
        categories: attached to an object crossing the scope boundary, retained
        by a sink process, or accumulated in a process's stock.

        :raises AssertionError: If conservation fails for any slice beyond
            `atol`, or if any slice's residual is undefined.
        :return: True if conservation holds for all slices.

        """
        residuals = self.meta["conservation_residuals"]
        undefined = sorted(k for k, v in residuals.items() if np.isnan(v))
        if undefined:
            raise AssertionError(
                f"Conservation is undefined for slices: {undefined}"
            )
        bad = {k: v for k, v in residuals.items() if abs(v) > atol}
        if bad:
            raise AssertionError(f"Conservation failed for slices: {bad}")
        return True


class Allocation:
    """Solve attributional average burdens (mu, beta) at a numeric operating point.

    :param model: A built, evaluable model (e.g. SympyModel)
    :param values: Numeric parameter values at which to evaluate (the
        operating point theta)
    :param rule: A `Rule` splitting each process's burden across its
        outputs.
    :param scope: Optional Scope selecting participating processes and
        waste-object cut-off behaviour. Default: all processes in scope, no
        waste objects.
    :param characterise: Optional ``{name: {exchange_id: factor}}`` extra
        characterised slices, in addition to the default per-exchange
        breakdown (mu/beta always carry the full per-exchange breakdown).
    :param wastes: Object ids to treat as wastes throughout: they take no
        share of the burden of the process producing them, and carry no
        burden into the process consuming them. Shorthand for wrapping `rule`
        in `Excluding` and setting the matching `Scope`; pass `rule` and
        `scope` directly to state the two halves separately.

    Access results via `.result` (an AllocationResult).
    """

    def __init__(
        self,
        model,
        values: dict,
        rule: Rule,
        scope: Optional[Scope] = None,
        characterise: Optional[dict] = None,
        wastes: Optional[frozenset] = None,
    ):
        if wastes is not None:
            if scope is not None:
                raise ValueError(
                    "pass either wastes= or scope=, not both: wastes= sets the "
                    "scope's waste objects itself"
                )
            wastes = frozenset(wastes)
            rule = Excluding(wastes, rule)
            scope = Scope(waste_objects=wastes, waste_input_burden="cutoff")

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
        roles = _flow_roles(model.structure, self.scope, X, Y, S, U)

        T = np.array(
            [
                sum(roles.borne[i, j] for j in range(M) if in_scope[j])
                for i in range(N)
            ]
        )

        _validate(self.rule, model.structure)
        weights, sinks, cutoffs = _allocation_weights(
            self.rule, model.structure, roles, in_scope
        )

        zero_supply = [i for i in range(N) if T[i] == 0]

        slice_names = [exc.id for exc in exchanges] + list(self.characterise.keys())
        q_vectors = {}
        for e, exc in enumerate(exchanges):
            q = np.zeros(len(exchanges))
            q[e] = 1.0
            q_vectors[exc.id] = q
        for name, factors in self.characterise.items():
            q_vectors[name] = np.array([factors.get(exc.id, 0) for exc in exchanges])

        # Direct burden as a total per process, on the same basis as the flow
        # quantities.
        direct = (
            np.vstack([q_vectors[name] @ B for name in slice_names])
            if slice_names
            else np.zeros((0, M))
        ) * Y

        # M[i, k] = sum_{j bears i, in scope} (w_ij / T_i) * carried[k, j]
        # (slice-independent: solved once for all slices as multiple RHS).
        Mmat = np.zeros((N, N))
        C = np.full((N, len(slice_names)), np.nan)
        for i in range(N):
            if T[i] == 0:
                continue
            C[i, :] = 0.0
            for j in range(M):
                if not in_scope[j] or weights[i, j] == 0:
                    continue
                share = weights[i, j] / T[i]
                Mmat[i, :] += share * roles.carried[:, j]
                C[i, :] += share * direct[:, j]

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

        # The burden each process hands on (`out_burden`: its own emissions
        # plus what its bearers are charged for the objects it took in) and
        # the burden that reached it (`in_burden`). An object with no modelled
        # supply contributes zero, but these are tracked as "unresolved".
        resolved = np.nan_to_num(mu)
        out_burden = np.zeros((len(slice_names), M))
        in_burden = np.zeros((len(slice_names), M))
        unresolved = np.zeros(M, dtype=bool)
        for j in range(M):
            if not in_scope[j]:
                continue
            out_burden[:, j] = direct[:, j] + roles.carried[:, j] @ resolved
            in_burden[:, j] = roles.paid[:, j] @ resolved
            unresolved[j] = any(
                T[i] == 0 and roles.carried[i, j] != 0 for i in range(N)
            )

        # Reported per unit of the process's reference activity; a process
        # with none has no unit to report against.
        reference = roles.reference_activity
        beta = np.where(
            (in_scope & (reference != 0) & ~unresolved)[:, None],
            out_burden.T / np.where(reference != 0, reference, 1.0)[:, None],
            np.nan,
        )

        residuals = _check_conservation_residuals(
            processes, roles, T, in_scope, mu, out_burden, in_burden,
            direct, slice_names, sinks
        )

        object_intensities = pd.DataFrame(
            mu, index=[o.id for o in objects], columns=slice_names
        )
        process_intensities = pd.DataFrame(
            beta, index=[p.id for p in processes], columns=slice_names
        )

        sigma_rows = [
            (objects[i].id, processes[j].id, roles.borne[i, j] / T[i])
            for i in range(N)
            for j in range(M)
            if roles.borne[i, j] > 0 and in_scope[j] and T[i] > 0
        ]
        supply_shares = pd.DataFrame(sigma_rows, columns=["object", "process", "sigma"])

        meta = {
            "rule": self.rule,
            "scope": self.scope,
            "cutoffs": cutoffs,
            "sinks": sinks,
            "zero_supply_objects": [objects[i].id for i in zero_supply],
            "stock_burden": _stock_burden(
                processes, roles, in_scope, in_burden, out_burden, direct,
                slice_names
            ),
            "conservation_residuals": residuals,
        }

        return AllocationResult(
            object_intensities=object_intensities,
            process_intensities=process_intensities,
            supply_shares=supply_shares,
            meta=meta,
        )


@dataclass(frozen=True)
class _FlowRoles:
    """Which of each process's flows bear its burden, and which pay into it.

    Burden leaves a process along its *bearer* flows, split between them by the
    allocation rule, and enters along its *payer* flows, carrying the intensity
    of the object that flows. It is described in this way, rather than "inputs"
    and "outputs", so that the accounting will also work for processes whose
    determining flow is an input, like waste treatment.

    `paid` and `carried` describe the same payer flows twice, scaled to the
    input (X) and output (Y) activities of the process respectively. They differ
    only for processes with stock accumulation occurring at the operating point.
    `paid` is what the process actually took in, while `carried` is what
    downstream bearers are charged for. Its default is the payer flows scaled to
    the reference activity, which says the stock is transparent -- it holds and
    releases material at the intensity of what is flowing in. In principle it
    could be overriden to account for burdens linked to dynamic stock cohorts.

    The gap ``paid - carried`` is reported per process.

    :param bearers: Per process, the object ids bearing its burden.
    :param reference_activity: Per process, the activity its bearer flows are
        keyed to, and the basis its intensity is reported per unit of.
    :param borne: N x M quantity along each bearer flow.
    :param paid: N x M quantity along each payer flow, as taken in.
    :param carried: N x M payer quantity this period's bearers are charged for.

    """

    bearers: tuple
    reference_activity: np.ndarray
    borne: np.ndarray
    paid: np.ndarray
    carried: np.ndarray


def _flow_roles(structure, scope, X, Y, S, U) -> _FlowRoles:
    """Assign a role to each flow of each process.

    A process's outputs bear its burden, keyed to its output activity `Y`; its
    inputs pay into it, keyed to its input activity `X`. An input of a waste
    object under a cut-off `Scope` is ignored.
    """
    use = U.copy()
    if scope.waste_input_burden == "cutoff":
        for object_id in scope.waste_objects:
            use[structure.lookup_object(object_id), :] = 0.0
    return _FlowRoles(
        bearers=tuple(tuple(p.produces) for p in structure.processes),
        reference_activity=Y,
        borne=Y * S,
        paid=X * use,
        carried=Y * use,
    )


def _normalised_weights(rule, process, flows):
    """Weights summing to 1 for one process, or None if it is a sink.
    """
    if not flows:
        return None
    weights = rule.weights(process, flows)
    if weights.keys() != flows.keys():
        raise ValueError(
            f"{type(rule).__name__} returned weights for {sorted(weights)}, "
            f"but process {process.id!r} has flows {sorted(flows)}"
        )
    negative = sorted(k for k, w in weights.items() if w < 0)
    if negative:
        raise ValueError(
            f"{type(rule).__name__} gave process {process.id!r} negative "
            f"weights for {negative}"
        )
    total = sum(weights.values())
    if total == 0:
        return None
    return {object_id: w / total for object_id, w in weights.items()}


def _allocation_weights(rule, structure, roles, in_scope):
    """Ask `rule` to split every in-scope process's burden across its bearers.

    :return: ``(weights, sinks, cutoffs)`` -- an N x M array of normalised
        weights; the ids of processes whose burden is not further allocated;
        and ``(process_id, object_id)`` pairs given zero weight despite a
        nonzero output flow.
    """
    weights = np.zeros(roles.borne.shape)
    sinks, cutoffs = [], []

    for j, process in enumerate(structure.processes):
        if not in_scope[j]:
            continue
        index = {
            object_id: structure.lookup_object(object_id)
            for object_id in roles.bearers[j]
        }
        flows = {object_id: roles.borne[i, j] for object_id, i in index.items()}
        shares = _normalised_weights(rule, process, flows)
        if shares is None:
            sinks.append(process.id)
            continue
        for object_id, weight in shares.items():
            weights[index[object_id], j] = weight
            if weight == 0 and flows[object_id] > 0:
                cutoffs.append((process.id, object_id))

    return weights, sinks, cutoffs


def _stock_burden(processes, roles, in_scope, in_burden, out_burden, direct,
                  slice_names) -> pd.DataFrame:
    """Burden that went into (+) or came out of (-) each process's stock.

    Zero for every process taking in what it puts out, so only the processes
    accumulating or depleting a stock appear.
    """
    rows = {
        p.id: in_burden[:, j] - (out_burden[:, j] - direct[:, j])
        for j, p in enumerate(processes)
        if in_scope[j] and np.any(roles.paid[:, j] != roles.carried[:, j])
    }
    return pd.DataFrame(rows.values(), index=list(rows), columns=slice_names)


def _check_conservation_residuals(
    processes, roles, T, in_scope, mu, out_burden, in_burden, direct,
    slice_names, sinks
):
    """Per-slice: boundary output + sinks + stock accumulation - direct burden.

    Burden entering the system as a process's own emissions should balance via:

    - objects crossing the boundary (the consumption deficit within scope);
    - sink processes (with no outputs to allocate its burden to);
    - process stock accumulation.

    An object with no modelled supply is assumed to contribute zero, exactly as
    the `mu` solve treats it.

    """
    M = len(processes)
    N = len(T)
    sinks = set(sinks)
    residuals = {}

    paid_total = np.array(
        [
            sum(roles.paid[i, j] for j in range(M) if in_scope[j])
            for i in range(N)
        ]
    )

    for col, name in enumerate(slice_names):
        boundary = sum(
            mu[i, col] * (T[i] - paid_total[i]) for i in range(N) if T[i] != 0
        )
        retained = sum(
            out_burden[col, j]
            for j, p in enumerate(processes)
            if in_scope[j] and p.id in sinks
        )
        into_stock = sum(
            in_burden[col, j] - (out_burden[col, j] - direct[col, j])
            for j in range(M)
            if in_scope[j]
        )
        emitted = sum(direct[col, j] for j in range(M) if in_scope[j])
        residuals[name] = boundary + retained + into_stock - emitted

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
