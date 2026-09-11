"""The allocated system: solving average burdens at one operating point."""

import weakref
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import sympy as sy

from .contributions import ProcessSet, contributions_of
from .rules import Excluding, Rule, validate_rule

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


class AllocatedSystem:
    """A linear representation of a model at one operating point, with
    allocation rules applied.

    Building one evaluates the model at `values`, splits each process's burden
    across its flows using `rule`, and solves for the average burden carried by
    every object and process. The results are available as intensities -- per
    unit of an object, or per unit of a process's reference activity -- and as
    the total burdens at that operating point.

    To see where one of those burdens came from, use :meth:`contributions`.

    **Example**::

        allocated = AllocatedSystem(model, params, rule=ByValue())
        allocated.object_intensities.loc["Polymer"]
        allocated.check_conservation()

    :param model: A built, evaluable model (e.g. `SympyModel`).
    :param values: Numeric parameter values at which to evaluate it.
    :param rule: A `Rule` splitting each process's burden across its outputs.
    :param scope: Optional `Scope` selecting participating processes and
        waste-object cut-off behaviour. Default: all processes in scope, no
        waste objects.
    :param characterise: Optional ``{name: {exchange_id: factor}}`` extra
        characterised slices, in addition to the per-exchange breakdown that
        the intensities always carry.
    :param wastes: Object ids to treat as wastes throughout: they take no
        share of the burden of the process producing them, and carry no
        burden into the process consuming them. Shorthand for wrapping `rule`
        in `Excluding` and setting the matching `Scope`; pass `rule` and
        `scope` directly to state the two halves separately.

    The solve is reported through these attributes:

    :ivar objects: Object ids, in order.
    :ivar processes: Process ids, in order.
    :ivar slices: The burden slices -- one per elementary exchange, plus one
        per entry in `characterise`. These are the columns of every table
        below.
    :ivar object_intensities: Objects x slices. Burden carried by one unit of
        each object; NaN for an object with no modelled supply.
    :ivar process_intensities: Processes x slices. Burden leaving each process
        per unit of its reference activity; NaN where there is no activity to
        report against, or where the process depends on an object with no
        modelled supply.
    :ivar object_burdens: Objects x slices. Total burden carried by all of
        each object supplied at this operating point.
    :ivar process_burdens: Processes x slices. Total burden leaving each
        process: its own emissions, plus what it was charged for its inputs.
    :ivar direct_burdens: Processes x slices. Total burden each process itself
        emitted.
    :ivar supply_shares: Long-form ``(object, process, sigma)`` -- the share
        of each object's supply that bears each process's burden.
    :ivar stock_burden: Processes x slices, for the processes accumulating or
        depleting a stock: burden that went into (+) or came out of (-) it.
    :ivar sinks: Process ids whose burden has no outputs to be allocated to,
        and so is retained rather than passed on.
    :ivar cutoffs: ``(process_id, object_id)`` pairs given zero weight by the
        rule despite a nonzero output flow.
    :ivar unsupplied_objects: Object ids with no modelled supply in scope.
    :ivar conservation_residuals: Per slice, the residual of the 100% rule --
        see :meth:`check_conservation`.
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
        self._solve()

    def __repr__(self):
        return (
            f"AllocatedSystem({len(self.processes)} processes, "
            f"{len(self.objects)} objects, slices={list(self.slices)})"
        )

    def _solve(self):
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

        validate_rule(self.rule, model.structure)
        weights, sinks, cutoffs = _allocation_weights(
            self.rule, model.structure, roles, in_scope
        )

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

        object_ids = [o.id for o in objects]
        process_ids = [p.id for p in processes]
        in_scope_rows = np.where(in_scope[:, None], 1.0, np.nan)

        self.objects = tuple(object_ids)
        self.processes = tuple(process_ids)
        self.slices = tuple(slice_names)

        self.object_intensities = pd.DataFrame(
            mu, index=object_ids, columns=slice_names
        )
        self.process_intensities = pd.DataFrame(
            beta, index=process_ids, columns=slice_names
        )
        self.object_burdens = pd.DataFrame(
            resolved * T[:, None], index=object_ids, columns=slice_names
        )
        self.process_burdens = pd.DataFrame(
            out_burden.T * in_scope_rows, index=process_ids, columns=slice_names
        )
        self.direct_burdens = pd.DataFrame(
            direct.T * in_scope_rows, index=process_ids, columns=slice_names
        )
        self.supply_shares = pd.DataFrame(
            [
                (objects[i].id, processes[j].id, roles.borne[i, j] / T[i])
                for i in range(N)
                for j in range(M)
                if roles.borne[i, j] > 0 and in_scope[j] and T[i] > 0
            ],
            columns=["object", "process", "sigma"],
        )
        self.stock_burden = _stock_burden(
            processes, roles, in_scope, in_burden, out_burden, direct, slice_names
        )
        self.sinks = tuple(sinks)
        self.cutoffs = tuple(cutoffs)
        self.unsupplied_objects = tuple(
            objects[i].id for i in range(N) if T[i] == 0
        )
        self.conservation_residuals = _check_conservation_residuals(
            processes, roles, T, in_scope, mu, out_burden, in_burden,
            direct, slice_names, sinks
        )

        # The solve in array form, for contributions().
        self._in_scope = in_scope
        self._borne = roles.borne
        self._carried = roles.carried
        self._weights = weights
        self._total_borne = T
        self._reference_activity = reference
        self._direct = direct
        self._out_burden = out_burden

    def object_index(self, object_id: str) -> int:
        """Position of an object id, raising if it is unknown."""
        try:
            return self.objects.index(object_id)
        except ValueError:
            raise ValueError(f"Unknown object id {object_id}") from None

    def process_index(self, process_id: str) -> int:
        """Position of a process id, raising if it is unknown."""
        try:
            return self.processes.index(process_id)
        except ValueError:
            raise ValueError(f"Unknown process id {process_id}") from None

    def _share_per_unit(self) -> np.ndarray:
        """Objects x processes: the share of each process's total burden borne
        by one unit of each object. Zero for an object with no supply."""
        shares = np.zeros_like(self._weights)
        supplied = self._total_borne != 0
        shares[supplied, :] = (
            self._weights[supplied, :] / self._total_borne[supplied, None]
        )
        return shares

    def contributions(
        self,
        *,
        object: Optional[str] = None,
        process: Optional[str] = None,
        amount: float = 1.0,
        expand: ProcessSet,
    ):
        """Break one of this system's burdens down into named parts.

        The processes in `expand` are reported with their own direct burdens;
        everything they take in from elsewhere is collapsed into the
        cradle-to-gate burden of the input it came in as. See
        `flowprog.allocation.contributions` for what that means, and
        `ProcessSet` for how to choose the processes.

        What is broken down is one of:

        - ``object=`` -- `amount` units of that object, carrying the average
          burden of everything producing it;
        - ``process=`` -- `amount` units of that process's reference activity;
        - both -- `amount` units of that object *as produced by that process*,
          when the average across producers would hide the difference between
          routes.

        :param object: Object whose burden to break down.
        :param process: Process whose burden to break down.
        :param amount: How much of it, in the units the model uses.
        :param expand: The processes to report in detail. A `ProcessSet`, a
            process id, or an iterable of them.
        :return: A `ContributionResult`.
        """
        return contributions_of(
            self, object=object, process=process, amount=amount, expand=expand
        )

    def check_conservation(self, atol: float = 1e-6) -> bool:
        """Check the 100% rule for every slice.

        All burdens from the in-scope processes should be accounted for in
        three categories: attached to an object crossing the scope boundary,
        retained by a sink process, or accumulated in a process's stock.

        :param atol: Absolute tolerance per slice.
        :raises AssertionError: If conservation fails for any slice beyond
            `atol`, or if any slice's residual is undefined.
        :return: True if conservation holds for all slices.
        """
        residuals = self.conservation_residuals
        undefined = sorted(k for k, v in residuals.items() if np.isnan(v))
        if undefined:
            raise AssertionError(f"Conservation is undefined for slices: {undefined}")
        bad = {k: v for k, v in residuals.items() if abs(v) > atol}
        if bad:
            raise AssertionError(f"Conservation failed for slices: {bad}")
        return True


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
