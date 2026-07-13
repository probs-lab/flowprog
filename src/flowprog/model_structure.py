"""
ModelStructure: Immutable structure (processes, objects, symbols, lookups)
"""

from typing import Optional, Container
from dataclasses import dataclass
from rdflib import URIRef
import sympy as sy
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


@dataclass
class ElementaryExchange:
    """An ElementaryExchange represents a flow to or from the environment.

    :param id: Identifier for the exchange, e.g. "CO2", "CH4", "GHG_upstream_CO2e".
    :param metric: URI identifying the metric used to quantify this exchange.

    There is no per-Process list of exchanges (unlike `produces`/`consumes` for
    Objects); sparsity is implied by recipe data (the B matrix) instead.

    """

    id: str
    metric: URIRef


class ModelStructure:
    """Immutable model structure: processes, objects, symbols, and lookups.

    This class encapsulates the structural aspects of a model that don't change
    during building or evaluation.
    """

    def __init__(
        self,
        processes: list[Process],
        objects: list[Object],
        elementary_exchanges: list[ElementaryExchange] = (),
    ):
        """Initialize model structure.

        :param processes: List of processes in the model
        :param objects: List of objects (materials, products, energy) in the model
        :param elementary_exchanges: List of elementary exchanges (flows to/from
            the environment) declared for the model
        """
        # Store as tuples to emphasize immutability
        self.processes = tuple(processes)
        self.objects = tuple(objects)
        self.elementary_exchanges = tuple(elementary_exchanges)

        M = len(processes)
        N = len(objects)
        E = len(self.elementary_exchanges)

        # Build lookup indices
        self._obj_name_to_idx = {obj.id: i for i, obj in enumerate(objects)}
        self._process_name_to_idx = {proc.id: j for j, proc in enumerate(processes)}
        self._exchange_name_to_idx = {
            exc.id: e for e, exc in enumerate(self.elementary_exchanges)
        }

        # Build process-object connectivity
        processes_producing_object: dict[int, list[int]] = {}
        processes_consuming_object: dict[int, list[int]] = {}

        for j, p in enumerate(processes):
            if not p.consumes and not p.produces:
                raise ValueError(f"Process {p.id} has no recipe")

            for obj_name in p.produces:
                idx = self._obj_name_to_idx[obj_name]
                processes_producing_object.setdefault(idx, [])
                processes_producing_object[idx].append(j)

            for obj_name in p.consumes:
                idx = self._obj_name_to_idx[obj_name]
                processes_consuming_object.setdefault(idx, [])
                processes_consuming_object[idx].append(j)

        self._processes_producing_object = processes_producing_object
        self._processes_consuming_object = processes_consuming_object

        # Create symbolic matrices
        self.X = sy.IndexedBase("X", shape=(M,), nonnegative=True)
        self.Y = sy.IndexedBase("Y", shape=(M,), nonnegative=True)
        self.S = sy.IndexedBase("S", shape=(N, M), nonnegative=True)
        self.U = sy.IndexedBase("U", shape=(N, M), nonnegative=True)

        # Elementary exchange coefficients: signed, no nonnegative assumption
        # (LCA convention: positive = to environment, negative = from environment)
        self.B = sy.IndexedBase("B", shape=(E, M))

        # Structural symbols for object-level queries (resolved by compiler)
        self.Balance = sy.IndexedBase("Balance", shape=(N,))
        self.ProductionDeficit = sy.IndexedBase(
            "ProductionDeficit", shape=(N,), nonnegative=True
        )
        self.ConsumptionDeficit = sy.IndexedBase(
            "ConsumptionDeficit", shape=(N,), nonnegative=True
        )

        # Structural symbol for elementary-exchange queries (resolved by compiler).
        self.ElementaryBalance = sy.IndexedBase("ElementaryBalance", shape=(E,))

    def __repr__(self):
        return f"ModelStructure(processes={len(self.processes)}, objects={len(self.objects)})"

    def lookup_process(self, process_id: str) -> int:
        """Get process index from ID.

        :param process_id: Process identifier
        :return: Process index
        :raises ValueError: If process_id is unknown
        """
        if process_id in self._process_name_to_idx:
            return self._process_name_to_idx[process_id]
        raise ValueError(f"Unknown process id {process_id}")

    def lookup_object(self, object_id: str) -> int:
        """Get object index from ID.

        :param object_id: Object identifier
        :return: Object index
        :raises ValueError: If object_id is unknown
        """
        if object_id in self._obj_name_to_idx:
            return self._obj_name_to_idx[object_id]
        raise ValueError(f"Unknown object id {object_id}")

    def lookup_exchange(self, exchange_id: str) -> int:
        """Get elementary exchange index from ID.

        :param exchange_id: Elementary exchange identifier
        :return: Exchange index
        :raises ValueError: If exchange_id is unknown
        """
        if exchange_id in self._exchange_name_to_idx:
            return self._exchange_name_to_idx[exchange_id]
        raise ValueError(f"Unknown exchange id {exchange_id}")

    def producers_of(self, object_id: str) -> list[str]:
        """Get list of processes that produce an object.

        :param object_id: Object identifier
        :return: List of process IDs that produce this object
        """
        i = self.lookup_object(object_id)
        return [
            self.processes[j].id for j in self._processes_producing_object.get(i, [])
        ]

    def consumers_of(self, object_id: str) -> list[str]:
        """Get list of processes that consume an object.

        :param object_id: Object identifier
        :return: List of process IDs that consume this object
        """
        i = self.lookup_object(object_id)
        return [
            self.processes[j].id for j in self._processes_consuming_object.get(i, [])
        ]

    def expr(
        self,
        role: str,
        *,
        process_id: Optional[str] = None,
        object_id: Optional[str] = None,
        exchange_id: Optional[str] = None,
        limit_to_processes: Optional[Container[str]] = None,
    ) -> sy.Expr:
        """Build expression for a role (pure function, no state).

        :param role: One of "ProcessOutput", "ProcessInput", "SoldProduction",
            "Consumption", "ProcessElementaryFlow", "ElementaryFlows"
        :param process_id: Process identifier (required for ProcessOutput/ProcessInput/
            ProcessElementaryFlow)
        :param object_id: Object identifier
        :param exchange_id: Elementary exchange identifier (required for
            ProcessElementaryFlow/ElementaryFlows)
        :param limit_to_processes: Optional set of process IDs to limit to
        :return: Sympy expression for the role
        :raises ValueError: If role is unknown or required parameters missing
        """
        if role == "ProcessOutput":
            assert object_id is not None and process_id is not None
            i, j = self.lookup_object(object_id), self.lookup_process(process_id)
            return self.Y[j] * self.S[i, j]

        elif role == "ProcessInput":
            assert object_id is not None and process_id is not None
            i, j = self.lookup_object(object_id), self.lookup_process(process_id)
            return self.X[j] * self.U[i, j]

        elif role == "SoldProduction":
            assert object_id is not None
            i = self.lookup_object(object_id)
            pids = [j for j in self._processes_producing_object.get(i, [])]
            if limit_to_processes is not None:
                pids = [j for j in pids if self.processes[j].id in limit_to_processes]
            return sum(  # type: ignore
                (
                    self.expr(
                        "ProcessOutput",
                        process_id=self.processes[j].id,
                        object_id=object_id,
                    )
                    for j in pids
                ),
                sy.S.Zero,
            )

        elif role == "Consumption":
            assert object_id is not None
            i = self.lookup_object(object_id)
            pids = [j for j in self._processes_consuming_object.get(i, [])]
            if limit_to_processes is not None:
                pids = [j for j in pids if self.processes[j].id in limit_to_processes]
            return sum(  # type: ignore
                (
                    self.expr(
                        "ProcessInput",
                        process_id=self.processes[j].id,
                        object_id=object_id,
                    )
                    for j in pids
                ),
                sy.S.Zero,
            )

        elif role == "ProcessElementaryFlow":
            assert exchange_id is not None and process_id is not None
            e, j = self.lookup_exchange(exchange_id), self.lookup_process(process_id)
            return self.B[e, j] * self.Y[j]

        elif role == "ElementaryFlows":
            assert exchange_id is not None
            e = self.lookup_exchange(exchange_id)
            pids = range(len(self.processes))
            if limit_to_processes is not None:
                pids = [j for j in pids if self.processes[j].id in limit_to_processes]
            return sum(  # type: ignore
                (
                    self.expr(
                        "ProcessElementaryFlow",
                        exchange_id=exchange_id,
                        process_id=self.processes[j].id,
                    )
                    for j in pids
                ),
                sy.S.Zero,
            )

        else:
            raise ValueError(f"Unknown role {role!r}")

    def _compute_object_balance(self, object_id, values):
        """Compute (production - consumption) for an object from accumulated values."""
        i = self.lookup_object(object_id)
        flow_in = sum(
            (
                self.S[i, j] * values.get(self.Y[j], sy.S.Zero)
                for j in self._processes_producing_object.get(i, [])
            ),
            sy.S.Zero,
        )
        flow_out = sum(
            (
                self.U[i, j] * values.get(self.X[j], sy.S.Zero)
                for j in self._processes_consuming_object.get(i, [])
            ),
            sy.S.Zero,
        )
        return flow_in - flow_out

    def _compute_elementary_balance(self, exchange_id, values):
        """Compute Sum_j B[e, j] * Y[j] for an exchange from accumulated values."""
        e = self.lookup_exchange(exchange_id)
        return sum(
            (
                self.B[e, j] * values.get(self.Y[j], sy.S.Zero)
                for j in range(len(self.processes))
            ),
            sy.S.Zero,
        )

    def resolve_structural_symbols(self, expr, values):
        """Substitute structural symbols in an expression with their current values.

        Resolves: Balance[i], ProductionDeficit[i], ConsumptionDeficit[i],
        ElementaryBalance[e], X[j], Y[j]

        """
        if not isinstance(expr, sy.Basic):
            return expr

        subs = {}
        for sym in expr.atoms(sy.Indexed):
            if sym.base == self.X or sym.base == self.Y:
                subs[sym] = values.get(sym, sy.S.Zero)
            elif sym.base == self.Balance:
                idx = sym.indices[0]
                obj_id = self.objects[idx].id
                subs[sym] = self._compute_object_balance(obj_id, values)
            elif sym.base == self.ProductionDeficit:
                idx = sym.indices[0]
                obj_id = self.objects[idx].id
                balance = self._compute_object_balance(obj_id, values)
                subs[sym] = sy.Max(0, -balance, evaluate=False)
            elif sym.base == self.ConsumptionDeficit:
                idx = sym.indices[0]
                obj_id = self.objects[idx].id
                balance = self._compute_object_balance(obj_id, values)
                subs[sym] = sy.Max(0, balance, evaluate=False)
            elif sym.base == self.ElementaryBalance:
                idx = sym.indices[0]
                exchange_id = self.elementary_exchanges[idx].id
                subs[sym] = self._compute_elementary_balance(exchange_id, values)

        return expr.xreplace(subs) if subs else expr
