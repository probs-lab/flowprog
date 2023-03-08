from typing import Iterable, Optional
from dataclasses import dataclass, field
import sympy as sy
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


@dataclass
class Value:
    id: str
    value: Optional[float] = None
    history: list = field(default_factory=list)


@dataclass
class Model:
    processes: list[Process]
    objects: list[Object]
    X: dict[int, Value]
    Y: dict[int, Value]
    Z: dict[int, Value]
    # Delta: list[sy.Expr]
    S: dict[tuple[int, int], sy.Expr]
    U: dict[tuple[int, int], sy.Expr]
    # alpha: dict[tuple[int, int], sy.Expr]
    # beta: dict[tuple[int, int], sy.Expr]
    s: dict[tuple[int, int], sy.Expr]
    u: dict[tuple[int, int], sy.Expr]

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
                            (Y.value * S).subs(values),
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
                            (X.value * U).subs(values),
                        )
                    )
        return pd.DataFrame(rows, columns=["source", "target", "material", "value"])


def define_symbols(
    processes: list[Process],
    objects: list[Object],
    allocate_backwards: Optional[list]=None
) -> Model:
    M = len(processes)
    N = len(objects)
    obj_name_to_idx = {obj.id: i for i, obj in enumerate(objects)}
    processes_producing_object: dict[int, list[int]] = {}
    for j in range(M):
        for obj_name in processes[j].produces:
            idx = obj_name_to_idx[obj_name]
            processes_producing_object.setdefault(idx, [])
            processes_producing_object[idx] += [j]
    processes_consuming_object: dict[int, list[int]] = {}
    for j in range(M):
        for obj_name in processes[j].consumes:
            idx = obj_name_to_idx[obj_name]
            processes_consuming_object.setdefault(idx, [])
            processes_consuming_object[idx] += [j]

    X = {j: Value(f"X_{j}") for j in range(M)}
    Y = {j: Value(f"Y_{j}") for j in range(M)}
    Z = {
        i: Value(f"Z_{i}")
        for i in range(N)
        # if isinstance(objects[i], BalancedObject)
    }
    # Delta = {
    #     i: sy.Symbol(f"Delta_{i}")
    #     for i in range(N)
    #     if isinstance(objects[i], AccumulatingObject)
    # }
    S = {
        (i, j): sy.Symbol(f"S_{i}{j}")
        for j in range(M)
        for i in (obj_name_to_idx[name] for name in processes[j].produces)
    }
    U = {
        (i, j): sy.Symbol(f"U_{i}{j}")
        for j in range(M)
        for i in (obj_name_to_idx[name] for name in processes[j].consumes)
    }
    s = {(i, j): sy.Symbol(f"s_{i}{j}") for (i, j) in S.keys()}
    u = {(i, j): sy.Symbol(f"u_{i}{j}") for (i, j) in U.keys()}

    if allocate_backwards is None:
        allocate_backwards = []

    # # XXX should make the symbols be passed in by user?
    # alpha = {
    #     (i, j): sy.Symbol(f"alpha_{i}{j}")
    #     for i in range(N)
    #     for j in processes_producing_object.get(i, [])
    #     if (
    #         len(processes_producing_object.get(i, [])) > 1
    #         and (
    #             objects[i].allocate_backwards
    #             or objects[i].id in allocate_backwards
    #         )
    #     )
    # }
    # beta = {
    #     (i, j): sy.Symbol(f"beta_{i}{j}")
    #     for i in range(N)
    #     for j in processes_consuming_object.get(i, [])
    #     if (
    #         len(processes_consuming_object.get(i, [])) > 1
    #         and objects[i].allocate_forwards
    #     )
    # }

    return Model(processes, objects, X, Y, Z, S, U, s, u)


class ModelBuilder:
    def __init__(self, m: Model):
        self.m = m
        self._obj_name_to_idx = {obj.id: i for i, obj in enumerate(m.objects)}
        self._process_name_to_idx = {proc.id: j for j, proc in enumerate(m.processes)}

        processes_producing_object: dict[int, list[int]] = {}
        for j in range(len(m.processes)):
            for obj_name in m.processes[j].produces:
                idx = self._obj_name_to_idx[obj_name]
                processes_producing_object.setdefault(idx, [])
                processes_producing_object[idx] += [j]
        self._processes_producing_object = processes_producing_object

        processes_consuming_object: dict[int, list[int]] = {}
        for j in range(len(m.processes)):
            for obj_name in m.processes[j].consumes:
                idx = self._obj_name_to_idx[obj_name]
                processes_consuming_object.setdefault(idx, [])
                processes_consuming_object[idx] += [j]
        self._processes_consuming_object = processes_consuming_object

    def _lookup_process(self, process_id: str) -> int:
        if process_id in self._process_name_to_idx:
            return self._process_name_to_idx[process_id]
        else:
            raise ValueError(f"Unkown process id {process_id}")

    def _lookup_object(self, object_id: str) -> int:
        if object_id in self._obj_name_to_idx:
            return self._obj_name_to_idx[object_id]
        else:
            raise ValueError(f"Unkown object id {object_id}")

    def _set_value(
        self, value: Value, new_value: float, history_entry: str, existing="add"
    ):
        assert existing == "add"
        if value.value is None:
            value.value = 0.0
        value.value += new_value
        value.history += [history_entry]

    def pull_production(self, object_id: str, production_value, until_objects=None,
                        allocate_backwards=None):
        """Pull production value backwards through the model until `until_objects`."""

        if until_objects is None:
            until_objects = set()
        if allocate_backwards is None:
            allocate_backwards = {}

        # Save this value
        _log.debug(
            "pull_production: production of %s = %s", object_id, production_value
        )
        i = self._lookup_object(object_id)
        # self._set_value(self.m.Z[i], production_value, f"set production of {object_id} = {production_value}")

        # Pull through model
        processes = self._processes_producing_object.get(i, [])
        _log.debug("pull_production: producing processes: %s", processes)
        if len(processes) == 1:
            self.pull_process_output(
                self.m.processes[processes[0]].id,
                object_id,
                production_value,
                until_objects=until_objects,
                allocate_backwards=allocate_backwards,
            )
        elif len(processes) > 1:
            if object_id not in allocate_backwards:
                raise ValueError(
                    "Backwards allocation coefficient not defined for %s -> {%s}" %
                    (object_id, ", ".join(self.m.processes[j].id for j in processes))
                )
            alphas = allocate_backwards[object_id]
            # allocate
            for j in processes:
                pid = self.m.processes[j].id
                if pid not in alphas:
                    raise ValueError(
                        "Backwards allocation coefficient not defined for %s -> %s" %
                        (object_id, pid)
                    )
                alpha = alphas[pid]
                # TODO : clip to [0, 1]?
                output = production_value * alpha
                self.pull_process_output(
                    pid,
                    object_id,
                    output,
                    until_objects=until_objects,
                    allocate_backwards=allocate_backwards,
                )
        else:
            raise ValueError(f"No processes produce {object_id}")


    def push_consumption(self, object_id: str, consumption_value, until_objects=None):
        """Push consumption value forwards through the model until `until_objects`."""

        if until_objects is None:
            until_objects = set()

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
            self.push_process_input(
                self.m.processes[processes[0]].id,
                object_id,
                consumption_value,
                until_objects=until_objects,
            )
        elif len(processes) > 1:
            # allocate
            for j in processes:
                output = consumption_value * self.m.beta[i, j]
                self.push_process_input(
                    self.m.processes[j].id,
                    object_id,
                    output,
                    until_objects=until_objects,
                )
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

        # Save this value
        _log.debug(
            "pull_process_output: output of %s from %s = %s",
            object_id,
            process_id,
            value,
        )
        i = self._lookup_object(object_id)
        j = self._lookup_process(process_id)
        activity = value / self.m.S[i, j]

        self._set_value(
            self.m.Y[j],
            activity,
            f"set output of {object_id} from {process_id} = {value}",
        )

        # XXX did have condition  and self.m.X[j].value != self.m.Y[j].value:
        if not self.m.processes[j].has_stock:
            # Link to process input side NOTE activity is added -- so we don't
            # set X=Y here (that would be another way)
            self._set_value(
                self.m.X[j],
                activity,
                f"set output of {object_id} from {process_id} = {value}",
            )

        # Pull through model
        _log.debug(
            "pull_process_output: consumes objects: %s", self.m.processes[j].consumes
        )
        for obj in self.m.processes[j].consumes:
            if obj in until_objects:
                _log.debug("pull_process_output: reached %s, stopping", obj)
                continue
            i = self._lookup_object(obj)
            _log.debug("pull_process_output: object %s", self.m.objects[i])
            # XXX should this check be in pull_production?
            if not self.m.objects[i].has_market:
                _log.debug("pull_process_output: reached object without market %s, stopping", obj)
                continue
            production = activity * self.m.U[i, j]
            self.pull_production(obj, production, until_objects=until_objects,
                                 allocate_backwards=allocate_backwards)

    def push_process_input(
        self, process_id: str, object_id: str, value, until_objects=None
    ):
        """Specify process input forwards through the model until `until_objects`."""

        if until_objects is None:
            until_objects = set()

        # Save this value
        history = f"set input of {object_id} into {process_id} = {value}"
        _log.debug("push_process_input: %s", history)
        i = self._lookup_object(object_id)
        j = self._lookup_process(process_id)
        activity = value / self.m.U[i, j]

        self._set_value(self.m.X[j], activity, history)

        # XXX did have extra condition:  and self.m.X[j].value != self.m.Y[j].value:
        if not self.m.processes[j].has_stock:
            # Link to process output side
            # NOTE activity is added -- so we don't
            # set X=Y here (that would be another way)
            self._set_value(self.m.Y[j], activity, history)

        # Push through model
        _log.debug(
            "push_process_input: produces objects: %s", self.m.processes[j].produces
        )
        for obj in self.m.processes[j].produces:
            if obj in until_objects:
                _log.debug("push_process_input: reached %s, stopping", obj)
                continue
            i = self._lookup_object(obj)
            _log.debug("push_process_input: object %s", self.m.objects[i])
            # XXX should this check be in push_consumption?
            if not self.m.objects[i].has_market:
                _log.debug("push_process_input: reached object without market %s, stopping", obj)
                continue
            production = activity * self.m.S[i, j]
            self.push_consumption(obj, production, until_objects=until_objects)

    def balance_consumption(
        self,
        object_id: str,
        process_id: str,
        limit: Optional[list] = None,
    ):
        """Balance object `object_id` by adjusting process `process_id`.

        To dispatch following a merit order, call this successively.
        """

        if limit is None:
            limit = []

        i = self._lookup_object(object_id)
        flow_in = sum(
            self.m.S[i, j] * self.m.Y[j].value
            for j in self._processes_producing_object.get(i, [])
            if self.m.Y[j].value is not None
        )
        flow_out = sum(
            self.m.U[i, j] * self.m.X[j].value
            for j in self._processes_consuming_object.get(i, [])
            if self.m.X[j].value is not None
        )
        _log.debug(
            "balance_object: balance at %s: %s in, %s out", object_id, flow_in, flow_out
        )

        additional_consumption = sy.Max(0, flow_in - flow_out)

        j = self._lookup_process(process_id)
        if (i, j) not in self.m.U:
            _log.warning("cannot dispatch additional consumption of %s to process %s which does not consume it",
                         object_id, process_id)
            return
        if self.m.U[i, j] == 0:
            _log.warning("cannot dispatch additional consumption of %s to process %s with zero use coefficient",
                         object_id, process_id)
            return

        activity = additional_consumption / self.m.U[i, j]

        # XXX This would be better implemented by setting up a small linear
        # problem and then solving, and scaling down if the limit is reached?
        for limit_item in limit:
            if limit_item[0] == "consumption":
                raise NotImplementError()
                limit_value = self._find_flow_downstream_from_process(j, limit_item[0], limit_item[1])
                _log.debug("found limit value for %s: %s", limit_item, limit_value)
            else:
                raise NotImplementError()

            activity = sy.Min(limit_value, activity)

        history = f"additional consumption to balance {object_id}"
        # set means add here
        self._set_value(self.m.X[j], activity, history)
        if (
            not self.m.processes[j].has_stock
            and self.m.X[j].value != self.m.Y[j].value
        ):
            # Link to process output side
            self._set_value(self.m.Y[j], self.m.X[j].value, history)

    def balance_production(
        self,
        object_id: str,
        process_id: str,
        limit: Optional[list] = None,
    ):
        """Balance object `object_id` by adjusting process `process_id`.

        To dispatch following a merit order, call this successively.
        """

        if limit is None:
            limit = []
        # if consumption_merit_order is None:
        #     consumption_merit_order = []
        # if production_merit_order is None:
        #     production_merit_order = []

        i = self._lookup_object(object_id)
        flow_in = sum(
            self.m.S[i, j] * self.m.Y[j].value
            for j in self._processes_producing_object.get(i, [])
            if self.m.Y[j].value is not None
        )
        flow_out = sum(
            self.m.U[i, j] * self.m.X[j].value
            for j in self._processes_consuming_object.get(i, [])
            if self.m.X[j].value is not None
        )
        _log.debug(
            "balance_object: balance at %s: %s in, %s out", object_id, flow_in, flow_out
        )

        additional_production = sy.Max(0, flow_out - flow_in)
        # additional_consumption = sy.Max(0, flow_in - flow_out)

        # XXX no capacity limits -- just using first in order for now

        # for p in consumption_merit_order:
        #     j = self._lookup_process(p)
        #     if self.m.U[i, j] == 0:
        #         _log.warning("cannot dispatch additional consumption of %s to process %s with zero use coefficient",
        #                      object_id, p)
        #         continue
        #     activity = additional_consumption / self.m.U[i, j]
        #     history = f"additional consumption to balance {object_id}"
        #     # set means add here
        #     self._set_value(self.m.X[j], activity, history)
        #     if (
        #         not self.m.processes[j].has_stock
        #         and self.m.X[j].value != self.m.Y[j].value
        #     ):
        #         # Link to process output side
        #         self._set_value(self.m.Y[j], self.m.X[j].value, history)

        j = self._lookup_process(process_id)
        if (i, j) not in self.m.S:
            _log.warning("cannot dispatch additional production of %s to process %s which does not produce it",
                         object_id, process_id)
            return
        if self.m.S[i, j] == 0:
            _log.warning("cannot dispatch additional production of %s to process %s with zero supply coefficient",
                         object_id, process_id)
            return

        activity = additional_production / self.m.S[i, j]

        # XXX This would be better implemented by setting up a small linear
        # problem and then solving, and scaling down if the limit is reached?
        for limit_item in limit:
            if limit_item[0] == "production":
                limit_value = self._find_flow_upstream_from_process(j, limit_item[0], limit_item[1])
                _log.debug("found limit value for %s: %s", limit_item, limit_value)
            else:
                raise NotImplementError()

            activity = sy.Min(limit_value, activity)

        history = f"additional production to balance {object_id}"
        # set means add here
        self._set_value(self.m.Y[j], activity, history)
        if (
            not self.m.processes[j].has_stock
            and self.m.X[j].value != self.m.Y[j].value
        ):
            # Link to process input side
            self._set_value(self.m.X[j], self.m.Y[j].value, history)

    def _find_flow_upstream_from_object(self, i, flow_type, object_id):
        if flow_type != "production":
            raise NotImplementedError()
        i_target = self._lookup_object(object_id)
        if i == i_target:
            flow_in = sum(
                self.m.S[i, jj] * self.m.Y[jj].value
                for jj in self._processes_producing_object.get(i, [])
                if self.m.Y[jj].value is not None
            )
            _log.debug("find_flow_upstream: found %s --> %s", (flow_type, object_id), flow_in)
            return flow_in

        else:
            for jj in self._processes_producing_object.get(i, []):
                nested = self._find_flow_upstream_from_process(jj, flow_type, object_id)
                if nested is not None:
                    _log.debug("find_flow_upstream: found nested %s --> %s", self.m.processes[jj], nested)
                    return nested

        return None

    def _find_flow_upstream_from_process(self, j, flow_type, object_id):
        if flow_type != "production":
            raise NotImplementedError()
        result = 1
        for obj in self.m.processes[j].consumes:
            i = self._lookup_object(obj)
            nested = self._find_flow_upstream_from_object(i, flow_type, object_id)
            if nested is not None:
                nested *= self.m.U[i, j]
                _log.debug("find_flow_upstream: found nested %s into %s --> %s", obj, self.m.processes[j].id, nested)
                return nested

        return None

    def fill_blanks(self, fill_value):
        """Fill in any X and Y values that have not been set yet."""
        for j, value in self.m.X.items():
            if value.value is None:
                value.value = fill_value
        for j, value in self.m.Y.items():
            if value.value is None:
                value.value = fill_value


# def process_equations(j: int, m: Model, demand_driven = True) -> Iterable:
#     for i in range(len(m.objects)):
#         if (i, j) in m.s:
#             # yield sy.Eq(m.s[i, j], m.X[j] * m.S[i, j])
#             yield (m.s[i, j], -m.Y[j] * m.S[i, j])
#         if (i, j) in m.u:
#             # yield sy.Eq(m.u[i, j], m.X[j] * m.U[i, j])
#             yield (m.u[i, j], -m.X[j] * m.U[i, j])
#     # if demand_driven:
#     #     yield sy.Eq(m.X[j])

#     if not m.processes[j].has_stock:
#         # input and output side of process must balance
#         yield (m.X[j], -m.Y[j])


# def object_alloc_equations(i: int, m: Model) -> Iterable:
#     # if isinstance(m.objects[i], AccumulatingObject):
#     #     Delta = m.Delta[i]
#     # elif isinstance(m.objects[i], BalancedObject):
#     #     Delta = sy.S.Zero
#     # else:
#     #     return

#     # yield sy.Eq(m.Z[i], sum(m.u.get((i, j), sy.S.Zero) for j in range(len(m.X))) - Delta)

#     if m.objects[i].allocate_backwards:
#         # Equations fixing the link between input of object (Z) and supply flows
#         inputs = [(jj, a) for (ii, jj), a in m.s.items() if ii == i]
#         if inputs:
#             alphas = [(jj, a) for (ii, jj), a in m.alpha.items() if ii == i]
#             if len(inputs) == 1:
#                 # Don't include it as a variable where there is just one
#                 assert len(alphas) == 0
#                 alphas = [(inputs[0][0], 1.0)]
#             else:
#                 assert len(alphas) == len(inputs)
#             for j, a in alphas[:-1]:
#                 # yield sy.Eq(m.s[i, j], m.Z[j] * alpha)
#                 yield (m.s[i, j], -m.Z[i] * a)
#             j = alphas[-1][0]
#             yield (m.s[i, j], -m.Z[i] * (1 - sum(a for _, a in alphas[:-1])))
#     else:
#         supply_sum = sum(m.s.get((i, j), sy.S.Zero) for j in range(len(m.processes)))
#         if supply_sum.is_zero:
#             print("Warning: nothing produces object %s (%s); skipping supply constraint" % (i, m.objects[i].id))
#         else:
#             yield (m.Z[i], -supply_sum)

#     if m.objects[i].allocate_forwards:
#         raise NotImplementedError

#     else:
#         use_sum = sum(m.u.get((i, j), sy.S.Zero) for j in range(len(m.processes)))
#         if use_sum.is_zero:
#             print("Warning: nothing uses object %s (%s); skipping use constraint" % (i, m.objects[i].id))
#         else:
#             yield (m.Z[i], -use_sum)


# def build_matrix(terms, free_vars) -> sy.Matrix:
#     mat_rows = [] #sy.Matrix(len(terms), len(free_vars) + 1)
#     for i, row in enumerate(terms):
#         mat_row = [0] * len(free_vars)
#         #print(row)
#         for t in row:
#             #print("  ", t)
#             for j in range(len(free_vars)):
#                 mat_row[j] += t.expand().coeff(free_vars[j])
#             #print("    ", mat_row)
#         #print(sum([coeff * v for coeff, v in zip(mat_row, free_vars)]))
#         #print()
#         remaining = sum(row) - sum([coeff * v for coeff, v in zip(mat_row, free_vars)])
#         remaining = remaining.expand().simplify()
#         remaining_free_vars = set(remaining.atoms()) & set(free_vars)
#         if len(remaining_free_vars) > 0:
#             _log.error("Error building matrix row %d: %s", i, row)
#             raise NonlinearModelError("Nonlinear combination of free variables in residual: %s" % remaining)
#         mat_row.append(-remaining)
#         mat_rows.append(mat_row)
#     mat = sy.Matrix(mat_rows)
#     return mat


# def check_solution(m: Model, independent_vars, solution):
#     additional_independent_vars = set()
#     for k in (list(m.s.values()) + list(m.u.values())):
#         if k not in solution:
#             print("No solution found for", k)
#             continue
#         atoms = set(solution[k].atoms())
#         unsolved = atoms - set(independent_vars) - {-1}
#         if len(unsolved) > 0:
#             print("Undetermined solution for", k, ":", unsolved)
#             additional_independent_vars |= unsolved
#     return additional_independent_vars


# def choose_free_vars(m: Model, independent_vars: set) -> list[sy.Expr]:
#     return (
#         [v for v in m.s.values() if v not in independent_vars] +
#         [v for v in m.u.values() if v not in independent_vars] +
#         [v for v in m.X.values() if v not in independent_vars] +
#         [v for v in m.Y.values() if v not in independent_vars] +
#         [v for v in m.Z.values() if v not in independent_vars] +
#         [v for v in m.alpha.values() if v not in independent_vars]
#     )


# def solve(m: Model, independent_vars: list[sy.Expr]):
#     terms = (
#         [eq for j in range(len(m.processes)) for eq in process_equations(j, m)] +
#         [eq for i in range(len(m.objects)) for eq in object_alloc_equations(i, m)]
#     )
#     _log.debug("solving with independent variables: %s", independent_vars)
#     _log.debug("    terms: %s", terms)

#     sol = None
#     while sol is None and independent_vars:
#         free_vars = choose_free_vars(m, set(independent_vars))
#         _log.debug("    free_vars: %s", free_vars)
#         mat = build_matrix(terms, free_vars)
#         _log.debug("Mat:  %s", mat)
#         _log.debug("RREF: %s", mat.rref())
#         sol = sy.solve_linear_system(mat, *free_vars)
#         if sol is not None:
#             break
#         else:
#             independent_vars = independent_vars[:-1]
#             _log.debug("    no solution, trying again with fewer independent vars: %s", independent_vars)
#             # free_vars = free_vars[:-1]
#             # _log.debug("    no solution, trying again with free vars: %s", free_vars)

#     _log.debug("    solution: %s", sol)
#     additional_independent_vars = check_solution(m, independent_vars, sol)
#     _log.debug("    additional_vars: %s", additional_independent_vars)


#     return sol, independent_vars + list(additional_independent_vars)
