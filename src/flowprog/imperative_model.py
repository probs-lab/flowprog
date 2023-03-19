from typing import Iterable, Optional
from collections import Counter
from dataclasses import dataclass, field
import sympy as sy
from sympy import S
import pandas as pd
import logging


_log = logging.getLogger(__name__)


@dataclass
class Process:
    id: str
    produces: list[str]
    consumes: list[str]
    has_stock: bool = False


@dataclass
class Object:
    id: str
    has_market: bool = False


class Model:
    def __init__(self, processes: list[Process], objects: list[Object]):
        """Define a model containing `processes` and `objects`."""

        self.processes = processes
        self.objects = objects

        M = len(processes)
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

        self.X = {j: sy.Symbol(f"X_{j}") for j in range(M)}
        self.Y = {j: sy.Symbol(f"Y_{j}") for j in range(M)}
        self.S = {
            (i, j): sy.Symbol(f"S_{i},{j}")
            for j in range(M)
            for i in (self._obj_name_to_idx[name] for name in processes[j].produces)
        }
        self.U = {
            (i, j): sy.Symbol(f"U_{i},{j}")
            for j in range(M)
            for i in (self._obj_name_to_idx[name] for name in processes[j].consumes)
        }

        self._values: Counter[sy.Expr, sy.Expr] = Counter({})
        self._history: dict[sy.Expr, list[str]] = {}

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
        self, role: str, *, process: Optional[str] = None, object: Optional[str] = None
    ) -> sy.Expr:
        """Return expression for `role` (see PRObs Ontology)."""
        if role == "ProcessOutput":
            i, j = self._lookup_object(object), self._lookup_process(process)
            return self.Y[j] * self.S[i, j]
        elif role == "ProcessInput":
            i, j = self._lookup_object(object), self._lookup_process(process)
            return self.X[j] * self.U[i, j]
        elif role == "SoldProduction":
            i = self._lookup_object(object)
            return sum(
                self.expr("ProcessOutput", process=self.processes[j].id, object=object)
                for j in self._processes_producing_object.get(i, [])
            )
        elif role == "Consumption":
            i = self._lookup_object(object)
            return sum(
                self.expr("ProcessInput", process=self.processes[j].id, object=object)
                for j in self._processes_consuming_object.get(i, [])
            )
        else:
            raise ValueError("Unknown role %r" % role)

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
            if object_id not in allocate_backwards:
                raise ValueError(
                    "Backwards allocation coefficient not defined for %s -> {%s}" %
                    (object_id, ", ".join(self.processes[j].id for j in processes))
                )
            alphas = allocate_backwards[object_id]
            # allocate
            result = Counter()
            for j in processes:
                pid = self.processes[j].id
                if pid not in alphas:
                    raise ValueError(
                        "Backwards allocation coefficient not defined for %s -> %s" %
                        (object_id, pid)
                    )
                alpha = alphas[pid]
                # TODO : clip to [0, 1]?
                output = production_value * alpha
                this_result = self.pull_process_output(
                    pid,
                    object_id,
                    output,
                    until_objects=until_objects,
                    allocate_backwards=allocate_backwards,
                )
                result.update(this_result)
            return result
        else:
            raise ValueError(f"No processes produce {object_id}")

    def push_consumption(self, object_id: str, consumption_value, until_objects=None):
        """Push consumption value forwards through the model until `until_objects`."""

        if until_objects is None:
            until_objects = set()

        # Add the current object to `until_objects`, to avoid infinite loops
        until_objects = set(until_objects) | {object_id}

        # Save this value
        _log.debug(
            "push_consumption: consumption of %s = %s", object_id, consumption_value
        )
        i = self._lookup_object(object_id)
        # self._set_value(self.m.Z[i], consumption_value, f"set production of {object_id} = {consumption_value}")

        # Push through model
        processes = self._processes_consuming_object.get(i, [])
        _log.debug("push_consumption: consuming processes: %s", processes)
        if len(processes) == 1:
            return self.push_process_input(
                self.processes[processes[0]].id,
                object_id,
                consumption_value,
                until_objects=until_objects,
            )
        elif len(processes) > 1:
            # allocate
            result = Counter()
            for j in processes:
                output = consumption_value * self.beta[i, j]
                this_result = self.push_process_input(
                    self.processes[j].id,
                    object_id,
                    output,
                    until_objects=until_objects,
                )
                # merge sum
                result.update(this_result)
            return result
        else:
            raise ValueError(f"No processes consume {object_id}")

    def pull_process_output(
        self, process_id: str, object_id: str, value, until_objects=None,
        allocate_backwards=None,
    ):
        """Specify process output backwards through the model until `until_objects`."""

        if until_objects is None:
            until_objects = set()
        if allocate_backwards is None:
            allocate_backwards = {}

        # Add the current object to `until_objects`, to avoid infinite loops
        until_objects = set(until_objects) | {object_id}

        # Save this value
        _log.debug(
            "pull_process_output: output of %s from %s = %s",
            object_id,
            process_id,
            value,
        )
        i = self._lookup_object(object_id)
        j = self._lookup_process(process_id)
        activity = value / self.S[i, j]

        result = Counter({
            self.Y[j]: activity,
        })
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
            result.update(more)

        return result

    def push_process_input(
        self, process_id: str, object_id: str, value, until_objects=None
    ):
        """Specify process input forwards through the model until `until_objects`."""

        if until_objects is None:
            until_objects = set()

        # Add the current object to `until_objects`, to avoid infinite loops
        until_objects = set(until_objects) | {object_id}

        # Save this value
        history = f"set input of {object_id} into {process_id} = {value}"
        _log.debug("push_process_input: %s", history)
        i = self._lookup_object(object_id)
        j = self._lookup_process(process_id)
        activity = value / self.U[i, j]

        result = Counter({
            self.X[j]: activity,
        })
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
            more = self.push_consumption(obj, production, until_objects=until_objects)
            result.update(more)

        return result

    def object_balance(self, object_id: str) -> sy.Expr:
        """Return (production - consumption) for `object_id`."""
        i = self._lookup_object(object_id)
        flow_in = sum(
            self.S[i, j] * self[self.Y[j]]
            for j in self._processes_producing_object.get(i, [])
            if self[self.Y[j]] is not None
        )
        flow_out = sum(
            self.U[i, j] * self[self.X[j]]
            for j in self._processes_consuming_object.get(i, [])
            if self[self.X[j]] is not None
        )
        _log.debug(
            "balance_object: balance at %s: %s in, %s out", object_id, flow_in, flow_out
        )
        return flow_in - flow_out

    def object_production_deficit(self, object_id: str) -> sy.Expr:
        """Return Max(0, consumption - production) for `object_id`."""
        return sy.Max(0, -self.object_balance(object_id))

    def object_consumption_deficit(self, object_id: str) -> sy.Expr:
        """Return Max(0, production - consumption) for `object_id`."""
        return sy.Max(0, self.object_balance(object_id))

    def limit(self, values, expr, limit):
        """Scale down `values` as needed to avoid exceeding `limit` for `expr`."""
        # if symbol not in proposed:
        #     raise ValueError("Nothing proposed for %r" % symbol)
        limit = S(limit) \
            .subs({k: self[k] for k in self.X.values()}) \
            .subs({k: self[k] for k in self.Y.values()})
        current = S(expr) \
            .subs({k: self[k] for k in self.X.values()}) \
            .subs({k: self[k] for k in self.Y.values()})
        # S(self[symbol])
        proposed = S(expr).subs(values)
        return {
            k: sy.Piecewise(
                (S.Zero, current >= limit),
                (proposed, proposed <= limit - current),
                ((limit - current) / proposed * v, True),
            )
            for k, v in values.items()
        }

    # def fill_blanks(self, fill_value):
    #     """Fill in any X and Y values that have not been set yet."""
    #     for j, value in self.X.items():
    #         if value.value is None:
    #             value.value = fill_value
    #     for j, value in self.Y.items():
    #         if value.value is None:
    #             value.value = fill_value

    def add(self, *values, label=None):
        """Add `values` to model's symbols."""
        if label is None:
            # FIXME could use a better default label stored in `values`?
            label = "<unknown>"
        symbols = set()
        for v in values:
            self._values.update(v)
            symbols.update(v.keys())
        for symbol in symbols:
            self._history.setdefault(symbol, [])
            self._history[symbol].append(label)

    def get_history(self, symbol: sy.Expr) -> list[str]:
        """Return history list for `symbol`."""
        return self._history.get(symbol, [])

    def __getitem__(self, symbol: sy.Expr) -> sy.Expr:
        """Get value stored for `symbol`."""
        # FIXME what to return when nothing stored?
        return self._values.get(symbol, sy.S.Zero)

    def to_flows(self, values):
        """Return flows data frame with variables substituted by `values`."""
        rows = []
        for j, Y in self.Y.items():
            for (ii, jj), S in self.S.items():
                if jj == j:
                    rows.append(
                        (
                            self.processes[j].id,
                            self.objects[ii].id,
                            self.objects[ii].id,
                            (self[Y] * S).subs(values),
                        )
                    )
        for j, X in self.X.items():
            for (ii, jj), U in self.U.items():
                if jj == j:
                    rows.append(
                        (
                            self.objects[ii].id,
                            self.processes[j].id,
                            self.objects[ii].id,
                            (self[X] * U).subs(values),
                        )
                    )
        return pd.DataFrame(rows, columns=["source", "target", "material", "value"])


