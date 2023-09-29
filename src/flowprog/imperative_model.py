from typing import Iterable, Optional
from collections import defaultdict
from dataclasses import dataclass, field
from rdflib import URIRef
import sympy as sy
from sympy import S
import pandas as pd
import logging


_log = logging.getLogger(__name__)


@dataclass
class Process:
    """A Process represents an activity involving input and/or output Objects.

    :param id: Identifier for the process.
    :param produces: Identifiers of objects that this process produces.
    :param consumes: Identifiers of objects that this process consumes.
    :param has_stock: Whether objects can accumulate within this process.

    If `has_stock` is `True`, the model will ensure that when the output of the
    process is determined, the input to the process must be equal, and vice
    versa. Otherwise setting the output of the process does not automatically
    imply the value of inputs until the change in internal stock level is also
    specified.

    """
    id: str
    produces: list[str]
    consumes: list[str]
    has_stock: bool = False


@dataclass
class Object:
    """An Object represents a type of material, product, or energy.

    :param id: Identifier for the object.
    :param metric: URI identifying the metric used to quantify this object in the model.
    :param has_market: Whether supply and demand of this object should be balanced.

    If `has_market` is `True`, demand for the object will propagate through to
    processes in the model which can produce the object, and vice versa for
    supply. Otherwise the market for the object is assumed to be outside the
    model boundary.

    """
    id: str
    metric: URIRef
    has_market: bool = False


class Model:
    def __init__(self, processes: list[Process], objects: list[Object]):
        """Define a model containing `processes` and `objects`."""

        self.processes = processes
        self.objects = objects

        M = len(processes)
        N = len(objects)
        self._obj_name_to_idx = {obj.id: i for i, obj in enumerate(objects)}
        self._process_name_to_idx = {proc.id: j for j, proc in enumerate(processes)}

        processes_producing_object: dict[int, list[int]] = {}
        processes_consuming_object: dict[int, list[int]] = {}

        for j, p in enumerate(processes):
            if not p.consumes and not p.produces:
                raise ValueError("Process %s has no recipe" % p.id)

            for obj_name in processes[j].produces:
                idx = self._obj_name_to_idx[obj_name]
                processes_producing_object.setdefault(idx, [])
                processes_producing_object[idx] += [j]

            for obj_name in processes[j].consumes:
                idx = self._obj_name_to_idx[obj_name]
                processes_consuming_object.setdefault(idx, [])
                processes_consuming_object[idx] += [j]

        self._processes_producing_object = processes_producing_object
        self._processes_consuming_object = processes_consuming_object

        self.X: sy.IndexedBase = sy.IndexedBase("X", shape=(M,), nonnegative=True)
        self.Y: sy.IndexedBase = sy.IndexedBase("Y", shape=(M,), nonnegative=True)
        self.S: sy.IndexedBase = sy.IndexedBase("S", shape=(N, M), nonnegative=True)
        self.U: sy.IndexedBase = sy.IndexedBase("U", shape=(N, M), nonnegative=True)

        self._values: dict[sy.Expr, sy.Expr] = defaultdict(lambda: sy.S.Zero)
        self._history: dict[sy.Expr, list[str]] = {}

        # Keep track of intermediate symbols
        self._intermediates: list[tuple[sy.Expr, sy.Expr, str]] = []
        self._intermediate_symbols = sy.numbered_symbols()

    def __repr__(self):
        return f"Model(processes={repr(self.processes)}, objects={repr(self.objects)})"

    def _lookup_process(self, process_id: str) -> int:
        if process_id in self._process_name_to_idx:
            return self._process_name_to_idx[process_id]
        else:
            raise ValueError(f"Unknown process id {process_id}")

    def _lookup_object(self, object_id: str) -> int:
        if object_id in self._obj_name_to_idx:
            return self._obj_name_to_idx[object_id]
        else:
            raise ValueError(f"Unknown object id {object_id}")

    def producers_of(self, object_id: str) -> Iterable[str]:
        i = self._lookup_object(object_id)
        return [self.processes[j].id for j in self._processes_producing_object.get(i, [])]

    def consumers_of(self, object_id: str) -> Iterable[str]:
        i = self._lookup_object(object_id)
        return [self.processes[j].id for j in self._processes_consuming_object.get(i, [])]

    def expr(
        self, role: str, *, process_id: Optional[str] = None, object_id: Optional[str] = None
    ) -> sy.Expr:
        """Return expression for `role` (see PRObs Ontology)."""
        if role == "ProcessOutput":
            assert object_id is not None and process_id is not None
            i, j = self._lookup_object(object_id), self._lookup_process(process_id)
            return self.Y[j] * self.S[i, j]
        elif role == "ProcessInput":
            assert object_id is not None and process_id is not None
            i, j = self._lookup_object(object_id), self._lookup_process(process_id)
            return self.X[j] * self.U[i, j]
        elif role == "SoldProduction":
            assert object_id is not None
            i = self._lookup_object(object_id)
            return sum(
                self.expr("ProcessOutput", process_id=self.processes[j].id, object_id=object_id)  # type:ignore
                for j in self._processes_producing_object.get(i, [])
            )
        elif role == "Consumption":
            assert object_id is not None
            i = self._lookup_object(object_id)
            return sum(
                self.expr("ProcessInput", process_id=self.processes[j].id, object_id=object_id)  # type:ignore
                for j in self._processes_consuming_object.get(i, [])
            )
        else:
            raise ValueError("Unknown role %r" % role)

    def _get_allocation(self, allocation, object_id, processes, tol=0.01):
        """Check that alphas sum to 1."""

        if object_id not in allocation:
            raise ValueError(
                "Allocation coefficient not defined for %s -> {%s}" %
                (object_id, ", ".join(self.processes[j].id for j in processes))
            )
        alphas = allocation[object_id]

        total = sum(alphas.values())
        pids = [self.processes[j].id for j in processes]
        # Don't check for symbolic sum...
        missing_processes = [pid for pid in pids if pid not in alphas]
        if isinstance(total, (int, float)):
            if abs(total - 1) > tol:
                msg = (
                    "Backwards allocation coefficient for %s does not sum to 1 (actual sum: %.3f)." %
                    (object_id, total)
                )
                if missing_processes:
                    msg += " These processes have not been included: %s" % (", ".join(missing_processes))
                raise ValueError(msg)
        elif missing_processes:
            # If we can't test with real coefficients that they sum to 1, we
            # require all symbols to be given
            msg = (
                "Not all processes are included in allocation coefficients: %s" %
                (", ".join(missing_processes))
            )
            raise ValueError(msg)

        return alphas

    def pull_production(self, object_id: str, production_value, until_objects=None,
                        allocate_backwards=None):
        """Pull production value backwards through the model until `until_objects`."""

        if until_objects is None:
            until_objects = set()
        if allocate_backwards is None:
            allocate_backwards = {}

        # Add the current object to `until_objects`, to avoid infinite loops
        until_objects = set(until_objects) | {object_id}

        # Save this value
        _log.debug(
            "pull_production: production of %s = %s", object_id, production_value
        )
        i = self._lookup_object(object_id)

        # Pull through model
        processes = self._processes_producing_object.get(i, [])
        _log.debug("pull_production: producing processes: %s", processes)
        if len(processes) == 1:
            return self.pull_process_output(
                self.processes[processes[0]].id,
                object_id,
                production_value,
                until_objects=until_objects,
                allocate_backwards=allocate_backwards,
            )
        elif len(processes) > 1:
            # allocate
            alphas = self._get_allocation(allocate_backwards, object_id, processes)
            result = defaultdict(lambda: sy.S.Zero)
            for j in processes:
                pid = self.processes[j].id
                if pid in alphas:
                    alpha = alphas[pid]
                    if sy.S(alpha).is_zero:
                        continue
                    # TODO : clip to [0, 1]?
                    output = production_value * alpha
                    this_result = self.pull_process_output(
                        pid,
                        object_id,
                        output,
                        until_objects=until_objects,
                        allocate_backwards=allocate_backwards,
                    )
                    for k, v in this_result.items():
                        result[k] += v
            return result
        else:
            raise ValueError(f"No processes produce {object_id}")

    def push_consumption(self, object_id: str, consumption_value, until_objects=None,
                         allocate_forwards=None):
        """Push consumption value forwards through the model until `until_objects`."""

        if until_objects is None:
            until_objects = set()
        if allocate_forwards is None:
            allocate_forwards = {}

        # Add the current object to `until_objects`, to avoid infinite loops
        until_objects = set(until_objects) | {object_id}

        # Save this value
        _log.debug(
            "push_consumption: consumption of %s = %s", object_id, consumption_value
        )
        i = self._lookup_object(object_id)

        # Push through model
        processes = self._processes_consuming_object.get(i, [])
        _log.debug("push_consumption: consuming processes: %s", processes)
        if len(processes) == 1:
            return self.push_process_input(
                self.processes[processes[0]].id,
                object_id,
                consumption_value,
                until_objects=until_objects,
                allocate_forwards=allocate_forwards,
            )
        elif len(processes) > 1:
            # allocate
            betas = self._get_allocation(allocate_forwards, object_id, processes)
            result = defaultdict(lambda: sy.S.Zero)
            for j in processes:
                pid = self.processes[j].id
                if pid in betas:
                    beta = betas[pid]
                    if sy.S(beta).is_zero:
                        continue
                    # TODO : clip to [0, 1]?
                    output = consumption_value * beta
                    this_result = self.push_process_input(
                        self.processes[j].id,
                        object_id,
                        output,
                        until_objects=until_objects,
                        allocate_forwards=allocate_forwards,
                    )
                    for k, v in this_result.items():
                        result[k] += v
            return result
        else:
            raise ValueError(f"No processes consume {object_id}")

    def pull_process_output(
        self, process_id: str, object_id: str, value, until_objects=None,
        allocate_backwards=None,
    ):
        """Specify process output backwards through the model until `until_objects`.

        If `object_id` is None, set the process input magnitude $X_j$ directly.
        Otherwise set $S_{ij} X_j$.

        """

        if until_objects is None:
            until_objects = set()
        if allocate_backwards is None:
            allocate_backwards = {}

        _log.debug(
            "pull_process_output: output of %s from %s = %s",
            object_id,
            process_id,
            value,
        )

        # Add the current object to `until_objects`, to avoid infinite loops
        until_objects = set(until_objects) | {object_id}

        # Save this value to an intermediate symbol with a description
        value = self._create_intermediate(
            value,
            f"pull_process_output value of {object_id} from {process_id}",
        )

        # Calculate required process activity
        j = self._lookup_process(process_id)
        if object_id is not None:
            i = self._lookup_object(object_id)
            activity = value / self.S[i, j]
        else:
            activity = value

        result = defaultdict(lambda: sy.S.Zero)
        result[self.Y[j]] += activity
        # save comment? f"set output of {object_id} from {process_id} = {value}",

        # XXX did have condition  and self.m.X[j].value != self.m.Y[j].value:
        if not self.processes[j].has_stock:
            # Link to process input side NOTE activity is added -- so we don't
            # set X=Y here (that would be another way)
            result[self.X[j]] = activity

        # Pull through model
        _log.debug(
            "pull_process_output: consumes objects: %s", self.processes[j].consumes
        )
        for obj in self.processes[j].consumes:
            if obj in until_objects:
                _log.debug("pull_process_output: reached %s, stopping", obj)
                continue
            i = self._lookup_object(obj)
            _log.debug("pull_process_output: object %s", self.objects[i])
            # XXX should this check be in pull_production?
            if not self.objects[i].has_market:
                _log.debug("pull_process_output: reached object without market %s, stopping", obj)
                continue
            production = activity * self.U[i, j]
            more = self.pull_production(obj, production, until_objects=until_objects,
                                        allocate_backwards=allocate_backwards)
            # merge sum
            for k, v in more.items():
                result[k] += v

        return result

    def push_process_input(
        self, process_id: str, object_id: str, value, until_objects=None,
        allocate_forwards=None,
    ):
        """Specify process input forwards through the model until `until_objects`.

        If `object_id` is None, set the process input magnitude $X_j$ directly.
        Otherwise set $U_{ij} X_j$.

        """

        if until_objects is None:
            until_objects = set()
        if allocate_forwards is None:
            allocate_forwards = {}

        _log.debug(
            "push_process_input: input of %s into %s = %s",
            object_id,
            process_id,
            value,
        )

        # Add the current object to `until_objects`, to avoid infinite loops
        until_objects = set(until_objects) | {object_id}

        # Save this value to an intermediate symbol with a description
        value = self._create_intermediate(
            value,
            f"push_process_input value of {object_id} from {process_id}",
        )

        # Calculate required process activity
        j = self._lookup_process(process_id)
        if object_id is not None:
            i = self._lookup_object(object_id)
            activity = value / self.U[i, j]
        else:
            activity = value

        result = defaultdict(lambda: sy.S.Zero)
        result[self.X[j]] += activity
        # save comment? f"set output of {object_id} from {process_id} = {value}",

        # XXX did have condition  and self.m.X[j].value != self.m.Y[j].value:
        if not self.processes[j].has_stock:
            # Link to process input side NOTE activity is added -- so we don't
            # set X=Y here (that would be another way)
            result[self.Y[j]] = activity

        # Push through model
        _log.debug(
            "push_process_input: produces objects: %s", self.processes[j].produces
        )
        for obj in self.processes[j].produces:
            if obj in until_objects:
                _log.debug("push_process_input: reached %s, stopping", obj)
                continue
            i = self._lookup_object(obj)
            _log.debug("push_process_input: object %s", self.objects[i])
            # XXX should this check be in push_consumption?
            if not self.objects[i].has_market:
                _log.debug("push_process_input: reached object without market %s, stopping", obj)
                continue
            production = activity * self.S[i, j]
            more = self.push_consumption(obj, production, until_objects=until_objects,
                                         allocate_forwards=allocate_forwards)
            # merge sum
            for k, v in more.items():
                result[k] += v

        return result

    def object_balance(self, object_id: str) -> sy.Expr:
        """Return (production - consumption) for `object_id`."""
        i = self._lookup_object(object_id)
        flow_in: sy.Expr = sum(  # type:ignore
            self.S[i, j] * self._values[self.Y[j]]
            for j in self._processes_producing_object.get(i, [])
            if self._values[self.Y[j]] is not None
        )
        flow_out: sy.Expr = sum(  # type:ignore
            self.U[i, j] * self._values[self.X[j]]
            for j in self._processes_consuming_object.get(i, [])
            if self._values[self.X[j]] is not None
        )
        _log.debug(
            "balance_object: balance at %s: %s in, %s out", object_id, flow_in, flow_out
        )
        return flow_in - flow_out

    def object_production_deficit(self, object_id: str) -> sy.Expr:
        """Return Max(0, consumption - production) for `object_id`."""
        # XXX Is evaluate needed here? It's *much* faster without
        value = sy.Max(0, -self.object_balance(object_id), evaluate=False)
        return self._create_intermediate(value, f"object_production_deficit for {object_id}")

    def object_consumption_deficit(self, object_id: str) -> sy.Expr:
        """Return Max(0, production - consumption) for `object_id`."""
        # XXX Is evaluate needed here? It's *much* faster without
        value = sy.Max(0, self.object_balance(object_id), evaluate=False)
        return self._create_intermediate(value, f"object_consumption_deficit for {object_id}")

    def limit(self, values, expr, limit):
        """Scale down `values` as needed to avoid exceeding `limit` for `expr`."""
        # if symbol not in proposed:
        #     raise ValueError("Nothing proposed for %r" % symbol)
        M = len(self.processes)
        limit = S(limit) \
            .subs({self.X[j]: self._values[self.X[j]] for j in range(M)}) \
            .subs({self.Y[j]: self._values[self.Y[j]] for j in range(M)})
        current = S(expr) \
            .subs({self.X[j]: self._values[self.X[j]] for j in range(M)}) \
            .subs({self.Y[j]: self._values[self.Y[j]] for j in range(M)})
        # S(self[symbol])
        proposed = S(expr).subs(values)
        return {
            k: sy.Piecewise(
                (S.Zero, current >= limit),
                (v, proposed <= limit - current),
                ((limit - current) / proposed * v, True),
                # It can be very slow...
                evaluate=False,
            )
            for k, v in values.items()
        }

    def add(self, *values, label=None):
        """Add `values` to model's symbols."""
        if label is None:
            # FIXME could use a better default label stored in `values`?
            label = "<unknown>"
        symbols = set()
        for v in values:
            # Assign a new intermediate variable for each new value.
            # Potentially this could be optimised.
            for sym, new_value in v.items():
                self._values[sym] += new_value
                # self._values[sym] += self._create_intermediate(
                #     new_value,
                #     f"Intermediate symbol for {sym} within '{label}'"
                # )
            symbols.update(v.keys())
        for symbol in symbols:
            self._history.setdefault(symbol, [])
            self._history[symbol].append(label)

    def _create_intermediate(self, value: sy.Expr, description: str) -> sy.Expr:
        new_sym = next(self._intermediate_symbols)
        self._intermediates.append((new_sym, value, description))
        return new_sym

    def get_history(self, symbol: sy.Expr) -> list[str]:
        """Return history list for `symbol`."""
        return self._history.get(symbol, [])

    # def __getitem__(self, symbol: sy.Expr) -> sy.Expr:
    #     """Get value stored for `symbol`."""
    #     # FIXME what to return when nothing stored?
    #     return self._values.get(symbol, sy.S.Zero)

    def eval_intermediates(self, expr: sy.Expr, values=None):
        """Substitute in `values` to intermediate expressions and then flows."""
        if values is None:
            values = {}
        intermediates = [
            (sym, (
                sym_value.xreplace(values)
                if isinstance(sym_value, sy.Expr)
                else sym_value
            ))
            for sym, sym_value, _ in self._intermediates
        ]
        return expr.subs(intermediates[::-1])

    def eval(self, symbol: sy.Expr, values=None):
        """Substitute in `values` to intermediate expressions and then flows."""
        symbol_value = self._values[symbol]
        return self.eval_intermediates(symbol_value)

    def lambdify(self, values, data_for_intermediates):
        """Return function to evalute model."""

        # Substitute recipe in intermediates now
        # FIXME double substitution
        subexpressions = [
            (sym, expr.xreplace(data_for_intermediates).xreplace(data_for_intermediates))
            for sym, expr, _ in self._intermediates
        ]

        args = (set()
                .union(*(expr.free_symbols for expr in values))
                .union(*(expr.free_symbols for _, expr in subexpressions))
                .difference(sym for sym, _ in subexpressions))
        # Indexed objects return themselves (e.g. S[1, 1]) as a free symbol as
        # well as the base matrix (e.g. S)
        args = {x for x in args if not isinstance(x, sy.Indexed)}
        args = list(args)

        f = sy.lambdify(args, values, cse=lambda expr: (subexpressions, expr))

        str_args = [str(x) for x in args]
        def wrapper(data):
            relevant_data = {k: v for k, v in data.items() if k in str_args}
            return f(**relevant_data)
        return wrapper

    def to_flows(self, values):
        """Return flows data frame with variables substituted by `values`."""
        rows = []
        M = len(self.processes)
        for j in range(M):
            for i in (self._obj_name_to_idx[name] for name in self.processes[j].produces):
                rows.append(
                    (
                        self.processes[j].id,
                        self.objects[i].id,
                        self.objects[i].id,
                        self.objects[i].metric,
                        (self._values[self.Y[j]] * self.S[i, j]).xreplace(values),
                    )
                )
        for j in range(M):
            for i in (self._obj_name_to_idx[name] for name in self.processes[j].consumes):
                rows.append(
                    (
                        self.objects[i].id,
                        self.processes[j].id,
                        self.objects[i].id,
                        self.objects[i].metric,
                        (self._values[self.X[j]] * self.U[i, j]).xreplace(values),
                    )
                )
        return pd.DataFrame(rows, columns=["source", "target", "material", "metric", "value"])
