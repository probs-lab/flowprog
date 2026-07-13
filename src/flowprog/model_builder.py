"""
Refactored model architecture with clear separation of concerns.

- ModelStructure: Immutable structure (processes, objects, symbols, lookups)
- ModelBuilder: Mutable builder for constructing symbolic model logic
- Model: Immutable evaluable model with recipe data
"""

from typing import Optional, Container
from collections import defaultdict
from rdflib import URIRef
import sympy as sy
from sympy import S
import logging

from .model_structure import Process, Object, ElementaryExchange, ModelStructure
from .activities import (
    AdditionalActivity,
    Limit,
    Floor,
    create_intermediate,
    merge_activities,
)
from .backends.markdown import compile_markdown
from .backends.sympy import SympyModel


_log = logging.getLogger(__name__)


class ModelBuilder:
    """Build symbolic model logic step by step.

    ModelBuilder works entirely with symbolic expressions and requires no
    numerical recipe data. It accumulates model logic through pull/push operations
    and tracks the history of how values were built up.

    When building is complete, call .build(recipe_data) to create an evaluable Model.
    """

    @classmethod
    def from_structure(cls, structure) -> "ModelBuilder":
        return cls(
            structure.processes, structure.objects, structure.elementary_exchanges
        )

    def __init__(
        self,
        processes: list[Process],
        objects: list[Object],
        elementary_exchanges: list[ElementaryExchange] = (),
    ):
        """Initialize model builder.

        :param processes: List of processes in the model
        :param objects: List of objects in the model
        :param elementary_exchanges: List of elementary exchanges declared for the model
        """
        self.structure = ModelStructure(processes, objects, elementary_exchanges)

        # Intermediate symbol counter (shared across all activities)
        self._intermediate_counter = sy.numbered_symbols()

        # Step recording
        self._steps: list[AdditionalActivity] = []

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

    @property
    def B(self):
        """Get B (elementary exchange) symbols from structure."""
        return self.structure.B

    # Delegate lookups and queries to structure
    def _lookup_process(self, process_id: str) -> int:
        return self.structure.lookup_process(process_id)

    def _lookup_object(self, object_id: str) -> int:
        return self.structure.lookup_object(object_id)

    def _lookup_exchange(self, exchange_id: str) -> int:
        return self.structure.lookup_exchange(exchange_id)

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
        exchange_id: Optional[str] = None,
        limit_to_processes: Optional[Container[str]] = None,
    ) -> sy.Expr:
        """Build expression for role (delegates to structure)."""
        return self.structure.expr(
            role,
            process_id=process_id,
            object_id=object_id,
            exchange_id=exchange_id,
            limit_to_processes=limit_to_processes,
        )

    def get_history(self, symbol: sy.Expr) -> list[str]:
        """Get history of how this symbol's value was built up.

        :param symbol: Sympy symbol to get history for
        :return: List of labels describing how this symbol's value was constructed
        """
        return [
            step.description
            for step in self._steps
            if step.description and symbol in step.values
        ]

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
        """Pull production value backwards through the model until `until_objects`.

        Returns an AdditionalActivity.
        """
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
            activities = []
            for j in processes:
                pid = self.processes[j].id
                if pid in alphas:
                    alpha = alphas[pid]
                    if sy.S(alpha).is_zero:
                        continue
                    output = production_value * alpha
                    activities.append(
                        self.pull_process_output(
                            pid,
                            object_id,
                            output,
                            until_objects=until_objects,
                            allocate_backwards=allocate_backwards,
                        )
                    )
            return merge_activities(*activities)
        else:
            raise ValueError(f"No processes produce {object_id}")

    def push_consumption(
        self,
        object_id: str,
        consumption_value,
        until_objects=None,
        allocate_forwards=None,
    ):
        """Push consumption value forwards through the model until `until_objects`.

        Returns an AdditionalActivity.
        """
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
            activities = []
            for j in processes:
                pid = self.processes[j].id
                if pid in betas:
                    beta = betas[pid]
                    _log.debug("push_consumption: alloc %s to %s", beta, pid)
                    if sy.S(beta).is_zero:
                        continue
                    output = consumption_value * beta
                    activities.append(
                        self.push_process_input(
                            self.processes[j].id,
                            object_id,
                            output,
                            until_objects=until_objects,
                            allocate_forwards=allocate_forwards,
                        )
                    )
            return merge_activities(*activities)
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

        Returns an AdditionalActivity.
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
        intermediates: list[tuple[sy.Symbol, sy.Expr, str]] = []
        value = create_intermediate(
            intermediates,
            value,
            f"pull_process_output value of {object_id} from {process_id}",
            self._intermediate_counter,
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
            # merge
            for k, v in more.values.items():
                result[k] += v
            intermediates.extend(more.intermediates)

        return AdditionalActivity(
            values=dict(result),
            intermediates=intermediates,
        )

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

        Returns an AdditionalActivity.
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
        intermediates: list[tuple[sy.Symbol, sy.Expr, str]] = []
        value = create_intermediate(
            intermediates,
            value,
            f"push_process_input value of {object_id} from {process_id}",
            self._intermediate_counter,
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
            # merge
            for k, v in more.values.items():
                result[k] += v
            intermediates.extend(more.intermediates)

        return AdditionalActivity(
            values=dict(result),
            intermediates=intermediates,
        )

    def object_balance(self, object_id: str) -> sy.Expr:
        """Return structural symbol for (production - consumption) of `object_id`.

        The compiler resolves this against accumulated state.
        """
        i = self._lookup_object(object_id)
        return self.structure.Balance[i]

    def object_production_deficit(self, object_id: str) -> sy.Expr:
        """Return structural symbol for Max(0, consumption - production) of `object_id`.

        The compiler resolves this against accumulated state.
        """
        i = self._lookup_object(object_id)
        return self.structure.ProductionDeficit[i]

    def object_consumption_deficit(self, object_id: str) -> sy.Expr:
        """Return structural symbol for Max(0, production - consumption) of `object_id`.

        The compiler resolves this against accumulated state.
        """
        i = self._lookup_object(object_id)
        return self.structure.ConsumptionDeficit[i]

    def elementary_balance(self, exchange_id: str) -> sy.Expr:
        """Return structural symbol for the cumulative net flow of `exchange_id`.

        Resolves to Sum_j B[e, j] * Y[j] against accumulated state. There is
        deliberately no deficit counterpart: elementary exchanges never have a
        producer to be in deficit against.
        """
        e = self._lookup_exchange(exchange_id)
        return self.structure.ElementaryBalance[e]

    def limit(self, activity, expr, limit):
        """Apply a capacity limit transformation to an AdditionalActivity.

        Returns a new AdditionalActivity with a Limit transformation appended.
        The expression and limit_value contain raw X[j]/Y[j] and structural
        symbols (Balance, ProductionDeficit, etc.) that the compiler resolves.

        :param activity: AdditionalActivity (or bare dict for backward compat)
        :param expr: Expression to constrain
        :param limit: Upper bound
        """
        if isinstance(activity, dict):
            activity = AdditionalActivity(values=activity)

        return AdditionalActivity(
            values=activity.values,
            intermediates=activity.intermediates,
            transformations=activity.transformations
            + [
                Limit(
                    expression=S(expr),
                    limit_value=S(limit),
                )
            ],
            description=activity.description,
        )

    def floor(self, activity, expr, threshold):
        """Apply a minimum turndown transformation to an AdditionalActivity.

        Returns a new AdditionalActivity with a Floor transformation appended.
        If the proposed value of `expr` (after adding this step) would be below
        `threshold`, the entire step is zeroed out.

        This implements a "minimum turndown" constraint: either the process
        operates above the minimum level, or it doesn't operate at all.

        :param activity: AdditionalActivity (or bare dict for backward compat)
        :param expr: Expression to check against threshold
        :param threshold: Minimum acceptable value
        """
        if isinstance(activity, dict):
            activity = AdditionalActivity(values=activity)

        return AdditionalActivity(
            values=activity.values,
            intermediates=activity.intermediates,
            transformations=activity.transformations
            + [
                Floor(
                    expression=S(expr),
                    threshold=S(threshold),
                )
            ],
            description=activity.description,
        )

    def add(self, *values, label=None):
        """Record activities as model steps.

        Accepts AdditionalActivity objects or bare dicts (backward compat).
        """
        for v in values:
            if isinstance(v, dict):
                activity = AdditionalActivity(values=v)
            else:
                activity = v

            if label is not None and activity.description is None:
                activity = AdditionalActivity(
                    values=activity.values,
                    intermediates=activity.intermediates,
                    transformations=activity.transformations,
                    description=label,
                )

            self._steps.append(activity)

    def build(self, recipe_data=None) -> SympyModel:
        """Create evaluable Model from this builder.

        Convenience wrapper for SympyModel.from_steps()

        :param recipe_data: Recipe data in format: ``{process_id: {consumes: {object_id: value}, produces: {object_id: value}}}``
        :return: SympyModel with recipe data
        """
        return SympyModel.from_steps(self._steps, self.structure, recipe_data)

    def describe(self) -> str:
        """Return a Markdown-formatted description of the model and its steps.

        Useful for debugging and understanding the model-building process.

        :return: Markdown string

        **Example**::

            print(builder.describe())
        """
        return compile_markdown(self.structure, self._steps)

    def save(self, filepath: str, metadata: Optional[dict] = None):
        """Save the complete model state to a JSON file.

        This saves all information needed to recreate the model, including:
        - Model structure (processes and objects)
        - Logical steps (the sequence of AdditionalActivity objects)

        Symbolic expressions are serialized using SymPy's srepr() which produces
        canonical string representations that can be exactly reconstructed.

        :param filepath: Path to save the model (will be overwritten if exists)
        :param metadata: Optional dictionary of metadata to include (e.g., description, author)

        **Example**::

            builder.save("my_model.json", metadata={"description": "Energy model v1.0"})
        """
        import json
        from datetime import datetime
        from .activities import serialize_step

        # Build the data structure
        data = {
            "version": "1.2",
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
            "elementary_exchanges": [
                {
                    "id": e.id,
                    "metric": str(e.metric),  # Convert URIRef to string
                }
                for e in self.structure.elementary_exchanges
            ],
            "steps": [serialize_step(step) for step in self._steps],
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        _log.info(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "ModelBuilder":
        """Load a model from a JSON file created by ModelBuilder.save().

        This recreates the complete model state including logical steps,
        assigned values, intermediate symbols.

        The model is reconstructed by:
        1. Creating the model structure (processes and objects)
        2. Deserializing all symbolic expressions using sympify()
        3. Restoring logical steps (if present in v1.1+ files)

        :param filepath: Path to the saved model file
        :return: Reconstructed ModelBuilder instance

        **Example**::

            builder = ModelBuilder.load("my_model.json")
        """
        import json
        from .activities import deserialize_step

        with open(filepath) as f:
            data = json.load(f)

        version = data.get("version", "1.0")
        if version not in ("1.0", "1.1", "1.2"):
            _log.warning(
                f"Model file version {version} may not be compatible "
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

        elementary_exchanges = [
            ElementaryExchange(
                id=e["id"],
                metric=URIRef(e["metric"]),  # Convert string back to URIRef
            )
            for e in data.get("elementary_exchanges", [])
        ]

        # Create the model (this creates X, Y, S, U, B which get cached by SymPy)
        builder = cls(processes, objects, elementary_exchanges)

        # Prepare namespace for sympify - include the builder's IndexedBase objects
        # and structural query symbols so that deserialized expressions reference them
        namespace = {
            "X": builder.X,
            "Y": builder.Y,
            "S": builder.S,
            "U": builder.U,
            "B": builder.B,
            "Balance": builder.structure.Balance,
            "ProductionDeficit": builder.structure.ProductionDeficit,
            "ConsumptionDeficit": builder.structure.ConsumptionDeficit,
            "ElementaryBalance": builder.structure.ElementaryBalance,
        }

        # Restore logical steps (v1.1+)
        if "steps" in data:
            builder._steps = [
                deserialize_step(step_data, namespace) for step_data in data["steps"]
            ]

        _log.info(f"Model loaded from {filepath}")
        if "metadata" in data and data["metadata"]:
            _log.info(f"Metadata: {data['metadata']}")

        return builder
