"""
Refactored model architecture with clear separation of concerns.

- ModelStructure: Immutable structure (processes, objects, symbols, lookups)
- ModelBuilder: Mutable builder for constructing symbolic model logic
- Model: Immutable evaluable model with recipe data
"""

import hashlib
from typing import Iterable, Optional, Container, Union
from collections import defaultdict
from dataclasses import dataclass
from rdflib import URIRef
import sympy as sy
from sympy import S
import numpy as np
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


class ModelStructure:
    """Immutable model structure: processes, objects, symbols, and lookups.

    This class encapsulates the structural aspects of a model that don't change
    during building or evaluation.
    """

    def __init__(self, processes: list[Process], objects: list[Object]):
        """Initialize model structure.

        :param processes: List of processes in the model
        :param objects: List of objects (materials, products, energy) in the model
        """
        # Store as tuples to emphasize immutability
        self.processes = tuple(processes)
        self.objects = tuple(objects)

        M = len(processes)
        N = len(objects)

        # Build lookup indices
        self._obj_name_to_idx = {obj.id: i for i, obj in enumerate(objects)}
        self._process_name_to_idx = {proc.id: j for j, proc in enumerate(processes)}

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

    def producers_of(self, object_id: str) -> list[str]:
        """Get list of processes that produce an object.

        :param object_id: Object identifier
        :return: List of process IDs that produce this object
        """
        i = self.lookup_object(object_id)
        return [
            self.processes[j].id
            for j in self._processes_producing_object.get(i, [])
        ]

    def consumers_of(self, object_id: str) -> list[str]:
        """Get list of processes that consume an object.

        :param object_id: Object identifier
        :return: List of process IDs that consume this object
        """
        i = self.lookup_object(object_id)
        return [
            self.processes[j].id
            for j in self._processes_consuming_object.get(i, [])
        ]

    def expr(
        self,
        role: str,
        *,
        process_id: Optional[str] = None,
        object_id: Optional[str] = None,
        limit_to_processes: Optional[Container[str]] = None,
    ) -> sy.Expr:
        """Build expression for a role (pure function, no state).

        :param role: One of "ProcessOutput", "ProcessInput", "SoldProduction", "Consumption"
        :param process_id: Process identifier (required for ProcessOutput/ProcessInput)
        :param object_id: Object identifier
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
                self.expr(
                    "ProcessOutput",
                    process_id=self.processes[j].id,
                    object_id=object_id,
                )
                for j in pids
            )

        elif role == "Consumption":
            assert object_id is not None
            i = self.lookup_object(object_id)
            pids = [j for j in self._processes_consuming_object.get(i, [])]
            if limit_to_processes is not None:
                pids = [j for j in pids if self.processes[j].id in limit_to_processes]
            return sum(  # type: ignore
                self.expr(
                    "ProcessInput",
                    process_id=self.processes[j].id,
                    object_id=object_id,
                )
                for j in pids
            )

        else:
            raise ValueError(f"Unknown role {role!r}")


class ModelBuilder:
    """Build symbolic model logic step by step.

    ModelBuilder works entirely with symbolic expressions and requires no
    numerical recipe data. It accumulates model logic through pull/push operations
    and tracks the history of how values were built up.

    When building is complete, call .build(recipe_data) to create an evaluable Model.
    """

    def __init__(self, processes: list[Process], objects: list[Object]):
        """Initialize model builder.

        :param processes: List of processes in the model
        :param objects: List of objects in the model
        """
        self.structure = ModelStructure(processes, objects)

        # Building state (mutable)
        self._values: dict[sy.Expr, sy.Expr] = defaultdict(lambda: sy.S.Zero)
        self._history: dict[sy.Expr, list[str]] = {}
        self._intermediates: list[tuple[sy.Expr, sy.Expr, str]] = []
        self._intermediate_symbols = sy.numbered_symbols()

    def __repr__(self):
        return f"ModelBuilder(structure={self.structure})"

    # Convenience accessors (delegate to structure)
    @property
    def processes(self):
        """Get processes from structure."""
        return self.structure.processes

    @property
    def objects(self):
        """Get objects from structure."""
        return self.structure.objects

    @property
    def X(self):
        """Get X symbols from structure."""
        return self.structure.X

    @property
    def Y(self):
        """Get Y symbols from structure."""
        return self.structure.Y

    @property
    def S(self):
        """Get S symbols from structure."""
        return self.structure.S

    @property
    def U(self):
        """Get U symbols from structure."""
        return self.structure.U

    # Delegate lookups and queries to structure
    def _lookup_process(self, process_id: str) -> int:
        return self.structure.lookup_process(process_id)

    def _lookup_object(self, object_id: str) -> int:
        return self.structure.lookup_object(object_id)

    def producers_of(self, object_id: str) -> list[str]:
        return self.structure.producers_of(object_id)

    def consumers_of(self, object_id: str) -> list[str]:
        return self.structure.consumers_of(object_id)

    def expr(
        self,
        role: str,
        *,
        process_id: Optional[str] = None,
        object_id: Optional[str] = None,
        limit_to_processes: Optional[Container[str]] = None,
    ) -> sy.Expr:
        """Build expression for role (delegates to structure)."""
        return self.structure.expr(
            role,
            process_id=process_id,
            object_id=object_id,
            limit_to_processes=limit_to_processes,
        )

    def get_history(self, symbol: sy.Expr) -> list[str]:
        """Get history of how this symbol's value was built up.

        :param symbol: Sympy symbol to get history for
        :return: List of labels describing how this symbol's value was constructed
        """
        return self._history.get(symbol, [])

    def print_history(self, symbol: sy.Expr):
        """Print history of how this symbol was built (for debugging).

        :param symbol: Sympy symbol to print history for
        """
        history = self.get_history(symbol)
        if history:
            print(f"History for {symbol}:")
            for label in history:
                print(f"  - {label}")
        else:
            print(f"No history for {symbol}")

    # Helper method for allocation checking (from current Model class)
    def _get_allocation(self, allocation, object_id, processes, tol=0.01):
        """Check that allocation coefficients sum to 1."""
        if object_id not in allocation:
            raise ValueError(
                f"Allocation coefficient not defined for {object_id} -> "
                f"{{{', '.join(self.processes[j].id for j in processes)}}}"
            )

        alphas = allocation[object_id]

        # Raises error for unknown processes by side effect
        for k in alphas.keys():
            self._lookup_process(k)

        total = sum(alphas.values())
        pids = [self.processes[j].id for j in processes]
        missing_processes = [pid for pid in pids if pid not in alphas]

        # Don't check for symbolic sum...
        if isinstance(total, (int, float)):
            if abs(total - 1) > tol:
                msg = (
                    f"Allocation coefficient for {object_id} does not sum to 1 "
                    f"(actual sum: {total:.3f})."
                )
                if missing_processes:
                    msg += f" These processes have not been included: {', '.join(missing_processes)}"
                raise ValueError(msg)
        elif missing_processes:
            # If we can't test with real coefficients that they sum to 1, we
            # require all symbols to be given
            raise ValueError(
                f"Not all processes are included in allocation coefficients: "
                f"{', '.join(missing_processes)}"
            )

        return alphas

    # ========== Building Methods ==========

    def pull_production(
        self,
        object_id: str,
        production_value,
        until_objects=None,
        allocate_backwards=None,
    ):
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
        processes = self.structure._processes_producing_object.get(i, [])
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

    def push_consumption(
        self,
        object_id: str,
        consumption_value,
        until_objects=None,
        allocate_forwards=None,
    ):
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
        processes = self.structure._processes_consuming_object.get(i, [])
        _log.debug("push_consumption: consuming processes: %s", processes)
        if len(processes) == 1:
            process_id = self.processes[processes[0]].id
            if object_id in allocate_forwards:
                # with only one consuming process, it is not required to include
                # the coefficients -- but check them if they are given
                alphas = allocate_forwards[object_id]
                if process_id not in alphas:
                    raise ValueError(
                        f"Process {process_id}, which is the only consumer of {object_id}, "
                        f"should be included in allocation coefficients"
                    )
                elif alphas[process_id] != 1:
                    raise ValueError(
                        f"Process {process_id}, which is the only consumer of {object_id}, "
                        f"should have an allocation coefficient of 1, "
                        f"but it is set to {alphas[process_id]}"
                    )
                for k in alphas.keys():
                    if alphas[k] != 0 and not self._lookup_process(k):
                        # raises error by side effect
                        pass
            return self.push_process_input(
                process_id,
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
                    _log.debug("push_consumption: alloc %s to %s", beta, pid)
                    if sy.S(beta).is_zero:
                        continue
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
        self,
        process_id: str,
        object_id: str,
        value,
        until_objects=None,
        allocate_backwards=None,
    ):
        """Specify process output backwards through the model until `until_objects`.

        If `object_id` is None, set the process output magnitude $Y_j$ directly.
        Otherwise set $S_{ij} Y_j$.
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
            # Validate that the process actually produces this object
            if object_id not in self.processes[j].produces:
                raise ValueError(
                    f"Process '{process_id}' does not produce object '{object_id}'. "
                    f"This process produces: {self.processes[j].produces}"
                )
            activity = value / self.S[i, j]
        else:
            activity = value

        result = defaultdict(lambda: sy.S.Zero)
        result[self.Y[j]] += activity

        if not self.processes[j].has_stock:
            # Link to process input side
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
            if not self.objects[i].has_market:
                _log.debug(
                    "pull_process_output: reached object without market %s, stopping",
                    obj,
                )
                continue
            production = activity * self.U[i, j]
            more = self.pull_production(
                obj,
                production,
                until_objects=until_objects,
                allocate_backwards=allocate_backwards,
            )
            # merge sum
            for k, v in more.items():
                result[k] += v

        return result

    def push_process_input(
        self,
        process_id: str,
        object_id: str,
        value,
        until_objects=None,
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
            # Validate that the process actually consumes this object
            if object_id not in self.processes[j].consumes:
                raise ValueError(
                    f"Process '{process_id}' does not consume object '{object_id}'. "
                    f"This process consumes: {self.processes[j].consumes}"
                )
            activity = value / self.U[i, j]
        else:
            activity = value

        result = defaultdict(lambda: sy.S.Zero)
        result[self.X[j]] += activity

        if not self.processes[j].has_stock:
            # Link to process output side
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
            if not self.objects[i].has_market:
                _log.debug(
                    "push_process_input: reached object without market %s, stopping",
                    obj,
                )
                continue
            production = activity * self.S[i, j]
            more = self.push_consumption(
                obj,
                production,
                until_objects=until_objects,
                allocate_forwards=allocate_forwards,
            )
            # merge sum
            for k, v in more.items():
                result[k] += v

        return result

    def object_balance(self, object_id: str) -> sy.Expr:
        """Return (production - consumption) for `object_id`."""
        i = self._lookup_object(object_id)
        flow_in: sy.Expr = sum(  # type: ignore
            self.S[i, j] * self._values[self.Y[j]]
            for j in self.structure._processes_producing_object.get(i, [])
            if self._values[self.Y[j]] is not None
        )
        flow_out: sy.Expr = sum(  # type: ignore
            self.U[i, j] * self._values[self.X[j]]
            for j in self.structure._processes_consuming_object.get(i, [])
            if self._values[self.X[j]] is not None
        )
        _log.debug(
            "balance_object: balance at %s: %s in, %s out", object_id, flow_in, flow_out
        )
        return flow_in - flow_out

    def object_production_deficit(self, object_id: str) -> sy.Expr:
        """Return Max(0, consumption - production) for `object_id`."""
        value = sy.Max(0, -self.object_balance(object_id), evaluate=False)
        return self._create_intermediate(
            value, f"object_production_deficit for {object_id}"
        )

    def object_consumption_deficit(self, object_id: str) -> sy.Expr:
        """Return Max(0, production - consumption) for `object_id`."""
        value = sy.Max(0, self.object_balance(object_id), evaluate=False)
        return self._create_intermediate(
            value, f"object_consumption_deficit for {object_id}"
        )

    def limit(self, values, expr, limit):
        """Scale down `values` as needed to avoid exceeding `limit` for `expr`.

        Note: This function assumes that process coefficients (S and U matrices)
        are non-zero. Zero coefficients will cause division by zero errors when
        expressions are evaluated.
        """
        M = len(self.processes)
        limit = (
            S(limit)
            .subs({self.X[j]: self._values[self.X[j]] for j in range(M)})
            .subs({self.Y[j]: self._values[self.Y[j]] for j in range(M)})
        )
        current = (
            S(expr)
            .subs({self.X[j]: self._values[self.X[j]] for j in range(M)})
            .subs({self.Y[j]: self._values[self.Y[j]] for j in range(M)})
        )
        proposed = (
            S(expr)
            .subs({self.X[j]: self._values[self.X[j]] + values.get(self.X[j], 0) for j in range(M)})
            .subs({self.Y[j]: self._values[self.Y[j]] + values.get(self.Y[j], 0) for j in range(M)})
        )

        # Calculate the difference
        diff = proposed - current

        # Protect against division by zero when lambdified to numpy.
        # When lambdify() converts Piecewise to numpy.select(), numpy evaluates
        # ALL branches before selecting. To prevent division by zero, we use
        # Max with a tiny epsilon to ensure the denominator is never exactly zero.
        # We use epsilon = 1e-10 which is small enough to not affect real values
        # but large enough for numerical stability.
        epsilon = S(10) ** -10  # Small value to prevent division by zero
        safe_diff = sy.Max(diff, epsilon)

        return {
            k: sy.Piecewise(
                # Already exceeded the limit, don't add any more
                (S.Zero, current >= limit),
                # New proposed value is less than or equal to the limit, no modification needed
                (v, proposed <= limit),
                # Proposed value needs to be scaled down to just reach limit
                # Use safe_diff (with epsilon protection) to avoid division by zero when lambdified
                ((limit - current) / safe_diff * v, True),
                evaluate=False,
            )
            for k, v in values.items()
        }

    def add(self, *values, label=None):
        """Add `values` to model's symbols."""
        if label is None:
            label = "<unknown>"
        symbols = set()
        for v in values:
            for sym, new_value in v.items():
                self._values[sym] += new_value
            symbols.update(v.keys())
        for symbol in symbols:
            self._history.setdefault(symbol, [])
            self._history[symbol].append(label)

    def _create_intermediate(self, value: sy.Expr, description: str) -> sy.Expr:
        """Create intermediate symbol for value."""
        new_sym = next(self._intermediate_symbols)
        self._intermediates.append((new_sym, value, description))
        return new_sym

    def eval_intermediates(self, expr: sy.Expr, values=None):
        """Substitute in `values` to intermediate expressions and then flows."""
        if values is None:
            values = {}
        intermediates = [
            (
                sym,
                (
                    sym_value.xreplace(values)
                    if isinstance(sym_value, sy.Expr)
                    else sym_value
                ),
            )
            for sym, sym_value, _ in self._intermediates
        ]
        return expr.subs(intermediates[::-1])

    def eval(self, symbol: sy.Expr, values=None):
        """Substitute in `values` to intermediate expressions and then flows."""
        if not isinstance(symbol, sy.Indexed) or symbol.base not in (self.X, self.Y):
            raise ValueError(f"symbol must be X[i] or Y[i]: {symbol!r}")
        symbol_value = self._values[symbol]
        return self.eval_intermediates(symbol_value, values)

    def build(self, recipe_data=None) -> 'Model':
        """Create evaluable Model from this builder.

        :param recipe_data: Recipe data in format: ``{process_id: {consumes: {object_id: value}, produces: {object_id: value}}}``
        :return: Evaluable Model with recipe data
        """
        return Model(
            structure=self.structure,  # Share structure!
            values=self._values,
            intermediates=self._intermediates,
            recipe_data=recipe_data,
        )

    def save(self, filepath: str, metadata: Optional[dict] = None):
        """Save the complete model state to a JSON file.

        This saves all information needed to recreate the model without rerunning
        model.add() calls, including:
        - Model structure (processes and objects)
        - All assigned values (_values)
        - Intermediate symbols and expressions (_intermediates)
        - History of how values were assigned (_history)

        Symbolic expressions are serialized using SymPy's srepr() which produces
        canonical string representations that can be exactly reconstructed.

        :param filepath: Path to save the model (will be overwritten if exists)
        :param metadata: Optional dictionary of metadata to include (e.g., description, author)

        **Example**::

            builder.save("my_model.json", metadata={"description": "Energy model v1.0"})
        """
        import json
        from datetime import datetime

        # Build the data structure
        data = {
            "version": "1.0",
            "metadata": metadata or {},
            "saved_at": datetime.now().isoformat(),
            "processes": [
                {
                    "id": p.id,
                    "produces": p.produces,
                    "consumes": p.consumes,
                    "has_stock": p.has_stock,
                }
                for p in self.processes
            ],
            "objects": [
                {
                    "id": o.id,
                    "metric": str(o.metric),  # Convert URIRef to string
                    "has_market": o.has_market,
                }
                for o in self.objects
            ],
            "values": {
                sy.srepr(k): sy.srepr(v)
                for k, v in self._values.items()
                if v != sy.S.Zero  # Don't save default zero values
            },
            "intermediates": [
                {
                    "symbol": sy.srepr(sym),
                    "expr": sy.srepr(expr),
                    "label": label,
                }
                for sym, expr, label in self._intermediates
            ],
            "history": {
                sy.srepr(k): v  # keys are expressions, values are already strings
                for k, v in self._history.items()
            },
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        _log.info(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "ModelBuilder":
        """Load a model from a JSON file created by ModelBuilder.save().

        This recreates the complete model state including all assigned values,
        intermediate symbols, and history.

        The model is reconstructed by:
        1. Creating the model structure (processes and objects)
        2. Deserializing all symbolic expressions using sympify()
        3. Restoring values, intermediates, and history

        Note: Due to SymPy's internal caching, IndexedBase objects (X, Y, S, U)
        created during model initialization will be automatically used by the
        deserialized expressions.

        :param filepath: Path to the saved model file
        :return: Reconstructed ModelBuilder instance

        **Example**::

            builder = ModelBuilder.load("my_model.json")
        """
        import json

        with open(filepath) as f:
            data = json.load(f)

        # Check version compatibility
        if data.get("version") != "1.0":
            _log.warning(
                f"Model file version {data.get('version')} may not be compatible "
                "with this version of flowprog"
            )

        # Reconstruct processes and objects
        processes = [
            Process(
                id=p["id"],
                produces=p["produces"],
                consumes=p["consumes"],
                has_stock=p["has_stock"],
            )
            for p in data["processes"]
        ]

        objects = [
            Object(
                id=o["id"],
                metric=URIRef(o["metric"]),  # Convert string back to URIRef
                has_market=o["has_market"],
            )
            for o in data["objects"]
        ]

        # Create the model (this creates X, Y, S, U which get cached by SymPy)
        builder = cls(processes, objects)

        # Prepare namespace for sympify - include the builder's IndexedBase objects
        # so that deserialized expressions reference them
        namespace = {
            'X': builder.X,
            'Y': builder.Y,
            'S': builder.S,
            'U': builder.U,
        }

        # Restore values
        builder._values = defaultdict(lambda: sy.S.Zero)
        for k_str, v_str in data["values"].items():
            try:
                k = sy.sympify(k_str, locals=namespace)
                v = sy.sympify(v_str, locals=namespace)
                builder._values[k] = v
            except Exception as e:
                _log.error(f"Failed to deserialize value {k_str}: {e}")
                raise

        # Restore intermediates
        builder._intermediates = []
        for item in data["intermediates"]:
            try:
                sym = sy.sympify(item["symbol"], locals=namespace)
                expr = sy.sympify(item["expr"], locals=namespace)
                builder._intermediates.append((sym, expr, item["label"]))
            except Exception as e:
                _log.error(f"Failed to deserialize intermediate {item['symbol']}: {e}")
                raise

        # Restore history
        builder._history = {}
        for k_str, v in data["history"].items():
            try:
                k = sy.sympify(k_str, locals=namespace)
                builder._history[k] = v
            except Exception as e:
                _log.error(f"Failed to deserialize history for {k_str}: {e}")
                raise

        _log.info(f"Model loaded from {filepath}")
        if "metadata" in data and data["metadata"]:
            _log.info(f"Metadata: {data['metadata']}")

        return builder


class Model:
    """Evaluable model with recipe data.

    Model represents a complete symbolic model with associated recipe data.
    It can be evaluated repeatedly with different parameter values. The model
    is immutable after creation (though recipe can be updated if needed).

    Models are created from ModelBuilder instances via builder.build(recipe_data).
    """

    def __init__(
        self,
        structure: ModelStructure,
        values: dict,
        intermediates: list,
        recipe_data=None,
    ):
        """Initialize evaluable model (typically called from ModelBuilder.build()).

        :param structure: Model structure (shared with builder)
        :param values: Symbolic expressions from builder
        :param intermediates: Intermediate symbols from builder
        :param recipe_data: Recipe data (optional)
        """
        self.structure = structure

        # Frozen symbolic expressions
        self._values = values
        self._intermediates = intermediates

        # Recipe data storage
        self._recipe_by_id: dict[str, dict] = {}
        self._recipe_cache: Optional[dict[sy.Indexed, Union[float, sy.Expr]]] = None

        if recipe_data:
            self.set_recipe(recipe_data)

    def __repr__(self):
        return f"Model(structure={self.structure}, has_recipe={bool(self._recipe_by_id)})"

    # Convenience accessors (delegate to structure)
    @property
    def processes(self):
        """Get processes from structure."""
        return self.structure.processes

    @property
    def objects(self):
        """Get objects from structure."""
        return self.structure.objects

    @property
    def X(self):
        """Get X symbols from structure."""
        return self.structure.X

    @property
    def Y(self):
        """Get Y symbols from structure."""
        return self.structure.Y

    @property
    def S(self):
        """Get S symbols from structure."""
        return self.structure.S

    @property
    def U(self):
        """Get U symbols from structure."""
        return self.structure.U

    def _lookup_process(self, process_id: str) -> int:
        return self.structure.lookup_process(process_id)

    def _lookup_object(self, object_id: str) -> int:
        return self.structure.lookup_object(object_id)

    def expr(
        self,
        role: str,
        *,
        process_id: Optional[str] = None,
        object_id: Optional[str] = None,
        limit_to_processes: Optional[Container[str]] = None,
    ) -> sy.Expr:
        """Build expression for role (delegates to structure)."""
        return self.structure.expr(
            role,
            process_id=process_id,
            object_id=object_id,
            limit_to_processes=limit_to_processes,
        )

    # ========== Recipe Management ==========

    def set_recipe(self, recipe_data):
        """Set recipe data for this model.

        :param recipe_data: Recipe in ontology-consistent format
            ``{process_id: {consumes: {object_id: value}, produces: {object_id: value}}}``
            OR backward-compatible symbol format
            ``{S[i, j]: value, U[i, j]: value}``
        """
        if not recipe_data:
            return

        # Detect format
        first_key = next(iter(recipe_data.keys()))

        if isinstance(first_key, sy.Indexed):
            # Symbol-based (backward compatible)
            self._set_recipe_from_symbols(recipe_data)
        else:
            # ID-based (new format)
            self._set_recipe_from_ids(recipe_data)

        # Invalidate cache
        self._recipe_cache = None

    def _set_recipe_from_symbols(self, recipe_data):
        """Convert symbol-based recipe to ID-based storage."""
        self._recipe_by_id = {}

        for symbol, value in recipe_data.items():
            if not isinstance(symbol, sy.Indexed):
                continue

            if symbol.base == self.U:
                i, j = symbol.indices
                obj_id = self.objects[i].id
                proc_id = self.processes[j].id

                if proc_id not in self._recipe_by_id:
                    self._recipe_by_id[proc_id] = {"consumes": {}, "produces": {}}
                self._recipe_by_id[proc_id]["consumes"][obj_id] = value

            elif symbol.base == self.S:
                i, j = symbol.indices
                obj_id = self.objects[i].id
                proc_id = self.processes[j].id

                if proc_id not in self._recipe_by_id:
                    self._recipe_by_id[proc_id] = {"consumes": {}, "produces": {}}
                self._recipe_by_id[proc_id]["produces"][obj_id] = value

    def _set_recipe_from_ids(self, recipe_data):
        """Set recipe from ID-based format with validation."""
        for proc_id, recipe in recipe_data.items():
            # Validate process exists
            j = self._lookup_process(proc_id)
            process = self.processes[j]

            # Validate consumed objects
            for obj_id in recipe.get("consumes", {}):
                if obj_id not in process.consumes:
                    raise ValueError(
                        f"Process '{proc_id}' recipe consumes '{obj_id}', "
                        f"but process definition only lists: {process.consumes}"
                    )

            # Validate produced objects
            for obj_id in recipe.get("produces", {}):
                if obj_id not in process.produces:
                    raise ValueError(
                        f"Process '{proc_id}' recipe produces '{obj_id}', "
                        f"but process definition only lists: {process.produces}"
                    )

            # Store recipe
            self._recipe_by_id[proc_id] = recipe

    def get_recipe(self, process_id: str) -> dict:
        """Get full recipe for a process in human-readable format.

        :param process_id: Process identifier
        :return: ``{"consumes": {obj_id: value}, "produces": {obj_id: value}}``
        """
        return self._recipe_by_id.get(process_id, {"consumes": {}, "produces": {}})

    def get_recipe_value(
        self, process_id: str, object_id: str, consumes: bool = True
    ) -> Optional[Union[float, sy.Expr]]:
        """Get a specific recipe value using IDs.

        :param process_id: Process identifier
        :param object_id: Object identifier
        :param consumes: True for consumption (U), False for production (S)
        :return: Recipe coefficient or None if not set
        """
        key = "consumes" if consumes else "produces"
        return self._recipe_by_id.get(process_id, {}).get(key, {}).get(object_id)

    def get_recipe_as_symbols(self) -> dict[sy.Indexed, Union[float, sy.Expr]]:
        """Get recipe data as symbol-based dict (cached for performance).

        :return: ``{U[i, j]: value, S[i, j]: value, ...}``
        """
        if self._recipe_cache is None:
            self._recipe_cache = {}

            for proc_id, recipe in self._recipe_by_id.items():
                j = self._lookup_process(proc_id)

                for obj_id, value in recipe.get("consumes", {}).items():
                    i = self._lookup_object(obj_id)
                    self._recipe_cache[self.U[i, j]] = value

                for obj_id, value in recipe.get("produces", {}).items():
                    i = self._lookup_object(obj_id)
                    self._recipe_cache[self.S[i, j]] = value

        return self._recipe_cache

    def print_recipe(self, process_id=None):
        """Print recipe in human-readable format.

        :param process_id: If provided, print only this process. Otherwise all.
        """
        if process_id:
            processes = [process_id]
        else:
            processes = self._recipe_by_id.keys()

        for proc_id in processes:
            recipe = self._recipe_by_id.get(proc_id, {})
            print(f"\n{proc_id}:")

            if recipe.get("consumes"):
                print("  Consumes:")
                for obj_id, value in recipe["consumes"].items():
                    print(f"    {obj_id}: {value}")

            if recipe.get("produces"):
                print("  Produces:")
                for obj_id, value in recipe["produces"].items():
                    print(f"    {obj_id}: {value}")

    # ========== Evaluation Methods ==========

    def eval_intermediates(self, expr: sy.Expr, values=None):
        """Substitute in `values` to intermediate expressions and then flows.

        :param expr: Expression to evaluate
        :param values: Additional values to substitute (recipe automatically included)
        :return: Evaluated expression
        """
        if values is None:
            values = {}

        # Merge recipe with provided values
        all_values = {**self.get_recipe_as_symbols(), **values}

        intermediates = [
            (
                sym,
                (
                    sym_value.xreplace(all_values)
                    if isinstance(sym_value, sy.Expr)
                    else sym_value
                ),
            )
            for sym, sym_value, _ in self._intermediates
        ]
        return expr.subs(intermediates[::-1])

    def eval(self, symbol: sy.Expr, values=None):
        """Evaluate a symbol with given values (recipe automatically included).

        :param symbol: Symbol to evaluate
        :param values: Additional values to substitute
        :return: Evaluated value
        """
        symbol_value = self._values[symbol]
        return self.eval_intermediates(symbol_value, values)

    def to_flows(self, values=None, flow_ids=None):
        """Return flows data frame with variables substituted.

        Recipe data is automatically included.

        :param values: Additional parameter values to substitute
        :param flow_ids: If True, assign hash-based flow ids to each row
        :return: DataFrame with flows
        """
        if values is None:
            values = {}

        # Merge recipe with provided values
        all_values = {**self.get_recipe_as_symbols(), **values}

        rows = []
        M = len(self.processes)

        for j in range(M):
            for i in (
                self.structure._obj_name_to_idx[name]
                for name in self.processes[j].produces
            ):
                expr = self._values[self.Y[j]] * self.S[i, j]
                # Resolve intermediates first, then substitute values
                resolved = self.eval_intermediates(expr, all_values)
                rows.append(
                    (
                        self.processes[j].id,
                        self.objects[i].id,
                        self.objects[i].id,
                        self.objects[i].metric,
                        resolved,
                    )
                )

        for j in range(M):
            for i in (
                self.structure._obj_name_to_idx[name]
                for name in self.processes[j].consumes
            ):
                expr = self._values[self.X[j]] * self.U[i, j]
                # Resolve intermediates first, then substitute values
                resolved = self.eval_intermediates(expr, all_values)
                rows.append(
                    (
                        self.objects[i].id,
                        self.processes[j].id,
                        self.objects[i].id,
                        self.objects[i].metric,
                        resolved,
                    )
                )

        result = pd.DataFrame(
            rows, columns=["source", "target", "material", "metric", "value"]
        )

        if flow_ids:

            def flow_id(row):
                return hashlib.md5(
                    (row["source"] + row["target"] + row["material"]).encode()
                ).hexdigest()

            result["id"] = result.apply(flow_id, axis=1)

        return result

    def lambdify(self, data=None, expressions: Optional[dict] = None):
        """Return function to evaluate model.

        Recipe data is automatically included (early substitution).

        :param data: Additional fixed data (beyond recipe) for early substitution
        :param expressions: Optional dict of expressions to evaluate.
                           If None, uses model flows.
        :return: Callable function that takes parameter dict and returns results
        """
        if data is None:
            data = {}

        # Merge recipe with additional data for early substitution
        all_data = {**self.get_recipe_as_symbols(), **data}

        if expressions is None:
            flows_sym = self.to_flows(all_data, flow_ids=True)
            index = flows_sym["id"]
            expr_values = flows_sym.value.values
        else:
            index = expressions.keys()
            expr_values = expressions.values()

        # Function that returns a vector of values in same order as index
        func = self._lambdify(expr_values, all_data)

        # Create a friendlier wrapper
        str_args = func.__code__.co_varnames[: func.__code__.co_argcount]

        def wrapper(data):
            converted_data = convert_indexed_symbols(data)
            relevant_data = {
                str(k): v for k, v in converted_data.items() if str(k) in str_args
            }
            missing_params = set(str_args) - set(relevant_data)
            if missing_params:
                raise ValueError(f"Missing parameters: {missing_params}")
            values = func(**relevant_data)
            # Convert to float if it's a 0-dimensional array
            values = [
                float(x) if isinstance(x, np.ndarray) and x.ndim == 0 else x
                for x in values
            ]
            return dict(zip(index, values))

        return wrapper

    def _lambdify(self, values, data_for_intermediates):
        """Internal lambdify implementation.

        :param values: Expressions to lambdify
        :param data_for_intermediates: Data to substitute into intermediates (early)
        :return: Lambdified function
        """
        # Substitute recipe/data in intermediates now (early substitution)
        subexpressions = [
            (
                sym,
                expr.xreplace(data_for_intermediates).xreplace(
                    data_for_intermediates
                ),
            )
            for sym, expr, _ in self._intermediates
        ]

        # Substitute data in values too
        values = [expr.xreplace(data_for_intermediates) for expr in values]

        # Find remaining free symbols (these become function parameters)
        args = (
            set()
            .union(*(expr.free_symbols for expr in values))
            .union(*(expr.free_symbols for _, expr in subexpressions))
            .difference(sym for sym, _ in subexpressions)
        )

        # Indexed objects return themselves (e.g. S[1, 1]) as a free symbol as
        # well as the base matrix (e.g. S) - filter these out
        args = {x for x in args if not isinstance(x, sy.Indexed)}
        args = list(args)

        f = sy.lambdify(args, values, cse=lambda expr: (subexpressions, expr))

        return f

    def save(self, filepath: str, metadata: Optional[dict] = None):
        """Save the complete model state including recipe to a JSON file.

        This saves all information needed to recreate the evaluable model:
        - Model structure (processes and objects)
        - All assigned values (_values)
        - Intermediate symbols and expressions (_intermediates)
        - Recipe data (both consumes and produces)

        Symbolic expressions are serialized using SymPy's srepr() which produces
        canonical string representations that can be exactly reconstructed.

        :param filepath: Path to save the model (will be overwritten if exists)
        :param metadata: Optional dictionary of metadata to include (e.g., description, author)

        **Example**::

            model.save("my_model.json", metadata={"description": "Energy model v1.0"})
        """
        import json
        from datetime import datetime

        # Build the data structure
        data = {
            "version": "1.0",
            "metadata": metadata or {},
            "saved_at": datetime.now().isoformat(),
            "type": "evaluable_model",  # Mark this as an evaluable model with recipe
            "processes": [
                {
                    "id": p.id,
                    "produces": p.produces,
                    "consumes": p.consumes,
                    "has_stock": p.has_stock,
                }
                for p in self.processes
            ],
            "objects": [
                {
                    "id": o.id,
                    "metric": str(o.metric),  # Convert URIRef to string
                    "has_market": o.has_market,
                }
                for o in self.objects
            ],
            "values": {
                sy.srepr(k): sy.srepr(v)
                for k, v in self._values.items()
                if v != sy.S.Zero  # Don't save default zero values
            },
            "intermediates": [
                {
                    "symbol": sy.srepr(sym),
                    "expr": sy.srepr(expr),
                    "label": label,
                }
                for sym, expr, label in self._intermediates
            ],
            "recipe": self._recipe_by_id,  # Save the ID-based recipe
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        _log.info(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "Model":
        """Load an evaluable model from a JSON file created by Model.save().

        This recreates the complete evaluable model state including recipe data.

        The model is reconstructed by:
        1. Creating the model structure (processes and objects)
        2. Deserializing all symbolic expressions using sympify()
        3. Restoring values and intermediates
        4. Restoring recipe data

        Note: Due to SymPy's internal caching, IndexedBase objects (X, Y, S, U)
        created during model initialization will be automatically used by the
        deserialized expressions.

        :param filepath: Path to the saved model file
        :return: Reconstructed Model instance (evaluable)

        **Example**::

            model = Model.load("my_model.json")
        """
        import json

        with open(filepath) as f:
            data = json.load(f)

        # Check version compatibility
        if data.get("version") != "1.0":
            _log.warning(
                f"Model file version {data.get('version')} may not be compatible "
                "with this version of flowprog"
            )

        # Check if this is an evaluable model or builder
        if data.get("type") != "evaluable_model":
            _log.warning(
                "This file was saved from a ModelBuilder, not an evaluable Model. "
                "Consider using ModelBuilder.load() instead."
            )

        # Reconstruct processes and objects
        processes = [
            Process(
                id=p["id"],
                produces=p["produces"],
                consumes=p["consumes"],
                has_stock=p["has_stock"],
            )
            for p in data["processes"]
        ]

        objects = [
            Object(
                id=o["id"],
                metric=URIRef(o["metric"]),  # Convert string back to URIRef
                has_market=o["has_market"],
            )
            for o in data["objects"]
        ]

        # Create structure
        structure = ModelStructure(processes, objects)

        # Prepare namespace for sympify
        namespace = {
            'X': structure.X,
            'Y': structure.Y,
            'S': structure.S,
            'U': structure.U,
        }

        # Restore values
        values = defaultdict(lambda: sy.S.Zero)
        for k_str, v_str in data["values"].items():
            try:
                k = sy.sympify(k_str, locals=namespace)
                v = sy.sympify(v_str, locals=namespace)
                values[k] = v
            except Exception as e:
                _log.error(f"Failed to deserialize value {k_str}: {e}")
                raise

        # Restore intermediates
        intermediates = []
        for item in data["intermediates"]:
            try:
                sym = sy.sympify(item["symbol"], locals=namespace)
                expr = sy.sympify(item["expr"], locals=namespace)
                intermediates.append((sym, expr, item["label"]))
            except Exception as e:
                _log.error(f"Failed to deserialize intermediate {item['symbol']}: {e}")
                raise

        # Create model with recipe
        model = cls(
            structure=structure,
            values=values,
            intermediates=intermediates,
            recipe_data=data.get("recipe"),  # Restore recipe
        )

        _log.info(f"Model loaded from {filepath}")
        if "metadata" in data and data["metadata"]:
            _log.info(f"Metadata: {data['metadata']}")

        return model


def convert_indexed_symbols(data):
    """Convert {S[1, 2]: 7} to {S: {(1, 2): 7}}

    This works better with lambdified functions.
    """
    converted = {}
    for k, v in data.items():
        if isinstance(k, sy.Indexed):
            sym = k.base
            if sym not in converted:
                converted[sym] = {}
            indices = k.indices if len(k.indices) > 1 else k.indices[0]
            converted[sym][indices] = v
        else:
            converted[k] = v
    return converted


# For backward compatibility during transition
# (Will be updated as we implement the full API)
