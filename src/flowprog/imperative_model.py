"""
Backward compatibility layer for the imperative Model API.

This module provides the same API as the original imperative_model.py,
but delegates all operations to the new ModelBuilder/Model architecture
in model.py.
"""

import hashlib
from typing import Iterable, Optional, Container
from collections import defaultdict
from dataclasses import dataclass
from rdflib import URIRef
import sympy as sy
from sympy import S
import numpy as np
import pandas as pd
import logging

# Import new implementation
from .model import (
    ModelBuilder as _ModelBuilder,
    Model as _Model,
    ModelStructure as _ModelStructure,
)

_log = logging.getLogger(__name__)


# Re-export Process and Object from model for consistency
from .model import Process, Object


class Model:
    """Backward compatibility wrapper for the new ModelBuilder/Model architecture.

    This class maintains the exact same API as the original Model class,
    but delegates all operations to ModelBuilder (for building) and Model
    (for evaluation) from model.py.

    Model instances are built fresh for each evaluation call to ensure
    recipe data changes are properly reflected.
    """

    def __init__(self, processes: list[Process], objects: list[Object]):
        """Define a model containing `processes` and `objects`."""
        # Create internal ModelBuilder to handle building operations
        self._builder = _ModelBuilder(processes, objects)
        self._intermediate_counter = sy.numbered_symbols()
        self._intermediates: list[tuple[sy.Expr, sy.Expr, str]] = []

    def __repr__(self):
        return f"Model(processes={repr(self.processes)}, objects={repr(self.objects)})"

    # Properties - delegate to builder/structure
    @property
    def processes(self):
        return self._builder.structure.processes

    @property
    def objects(self):
        return self._builder.structure.objects

    @property
    def X(self):
        return self._builder.X

    @property
    def Y(self):
        return self._builder.Y

    @property
    def S(self):
        return self._builder.S

    @property
    def U(self):
        return self._builder.U

    @property
    def _values(self):
        """Access to internal values for backward compatibility."""
        return self._builder._values

    @property
    def _history(self):
        """Access to internal history for backward compatibility."""
        return self._builder._history

    @property
    def _intermediate_symbols(self):
        """Access to internal intermediate symbols for backward compatibility."""
        return self._builder._intermediate_symbols

    @property
    def _obj_name_to_idx(self):
        """Access to object lookup for backward compatibility."""
        return self._builder.structure._obj_name_to_idx

    @property
    def _process_name_to_idx(self):
        """Access to process lookup for backward compatibility."""
        return self._builder.structure._process_name_to_idx

    @property
    def _processes_producing_object(self):
        """Access to process-object connectivity for backward compatibility."""
        return self._builder.structure._processes_producing_object

    @property
    def _processes_consuming_object(self):
        """Access to process-object connectivity for backward compatibility."""
        return self._builder.structure._processes_consuming_object

    # Lookup methods - delegate to structure
    def _lookup_process(self, process_id: str) -> int:
        return self._builder.structure.lookup_process(process_id)

    def _lookup_object(self, object_id: str) -> int:
        return self._builder.structure.lookup_object(object_id)

    def producers_of(self, object_id: str) -> Iterable[str]:
        return self._builder.structure.producers_of(object_id)

    def consumers_of(self, object_id: str) -> Iterable[str]:
        return self._builder.structure.consumers_of(object_id)

    def expr(
        self,
        role: str,
        *,
        process_id: Optional[str] = None,
        object_id: Optional[str] = None,
        limit_to_processes: Optional[Container[str]] = None,
    ) -> sy.Expr:
        """Return expression for `role` (see PRObs Ontology)."""
        return self._builder.structure.expr(
            role,
            process_id=process_id,
            object_id=object_id,
            limit_to_processes=limit_to_processes,
        )

    # Building methods - delegate to builder
    def _get_allocation(self, allocation, object_id, processes, tol=0.01):
        """Check that alphas sum to 1."""
        return self._builder._get_allocation(allocation, object_id, processes, tol)

    def pull_production(
        self,
        object_id: str,
        production_value,
        until_objects=None,
        allocate_backwards=None,
    ):
        """Pull production value backwards through the model until `until_objects`."""
        return self._builder.pull_production(
            object_id, production_value, until_objects, allocate_backwards
        )

    def push_consumption(
        self,
        object_id: str,
        consumption_value,
        until_objects=None,
        allocate_forwards=None,
    ):
        """Push consumption value forwards through the model until `until_objects`."""
        return self._builder.push_consumption(
            object_id, consumption_value, until_objects, allocate_forwards
        )

    def pull_process_output(
        self,
        process_id: str,
        object_id: str,
        value,
        until_objects=None,
        allocate_backwards=None,
    ):
        """Specify process output backwards through the model until `until_objects`."""
        return self._builder.pull_process_output(
            process_id, object_id, value, until_objects, allocate_backwards
        )

    def push_process_input(
        self,
        process_id: str,
        object_id: str,
        value,
        until_objects=None,
        allocate_forwards=None,
    ):
        """Specify process input forwards through the model until `until_objects`."""
        return self._builder.push_process_input(
            process_id, object_id, value, until_objects, allocate_forwards
        )

    def object_balance(self, object_id: str) -> sy.Expr:
        """Return (production - consumption) for `object_id`."""
        return self._builder.object_balance(object_id)

    def object_production_deficit(self, object_id: str) -> sy.Expr:
        """Return Max(0, consumption - production) for `object_id`."""
        return self._builder.object_production_deficit(object_id)

    def object_consumption_deficit(self, object_id: str) -> sy.Expr:
        """Return Max(0, production - consumption) for `object_id`."""
        return self._builder.object_consumption_deficit(object_id)

    def limit(self, values, expr, limit):
        """Scale down `values` as needed to avoid exceeding `limit` for `expr`."""
        return self._builder.limit(values, expr, limit)

    def add(self, *values, label=None):
        """Add `values` to model's symbols."""
        return self._builder.add(*values, label=label)

    # def _create_intermediate(self, value: sy.Expr, description: str) -> sy.Expr:
    #     """Create intermediate symbol for value."""
    #     new_sym = self._intermediate_counter.next()
    #     self._intermediates.append((new_sym, value, description))
    #     return new_sym

    def get_history(self, symbol: sy.Expr) -> list[str]:
        """Return history list for `symbol`."""
        return self._builder.get_history(symbol)

    # Evaluation methods - delegate to model (build fresh each time)
    def _extract_recipe_from_values(self, values):
        """Extract recipe data (S/U symbols) from values dict.

        Returns (recipe_data, remaining_values)
        """
        if values is None:
            return None, {}

        recipe_symbols = {}
        other_values = {}

        for k, v in values.items():
            # Check if this is a recipe symbol (S[i,j] or U[i,j])
            if isinstance(k, sy.Indexed) and k.base in (self.S, self.U):
                recipe_symbols[k] = v
            else:
                other_values[k] = v

        return recipe_symbols if recipe_symbols else None, other_values

    def eval_intermediates(self, expr: sy.Expr, values=None):
        """Substitute in `values` to intermediate expressions and then flows."""
        # Extract recipe data
        recipe_data, other_values = self._extract_recipe_from_values(values)

        # Build model with recipe (fresh each time)
        model = self._builder.build(recipe_data)

        # Resolve any structural symbols against compiled accumulated state
        expr = self._builder._resolve_structural_symbols(expr)

        # Delegate to model
        return model.eval_intermediates(expr, other_values)

    def eval(self, symbol: sy.Expr, values=None):
        """Substitute in `values` to intermediate expressions and then flows."""
        self._builder._compile()
        return self._builder.eval(symbol, values)

    def lambdify(self, data=None, expressions: Optional[dict] = None):
        """Return function to evalute model.

        If `expressions` is given as a dictionary of sympy expression, the
        resulting function returns a similar dict with evaluated values.
        Otherwise the model's flows are used as the expressions.
        """
        # Extract recipe from data
        recipe_data, other_data = self._extract_recipe_from_values(data)

        # Build model with recipe (fresh each time)
        model = self._builder.build(recipe_data)

        # Delegate to model
        return model.lambdify(other_data, expressions)

    def _lambdify(self, values, data_for_intermediates):
        """Return function to evalute model."""
        # This is an internal method used by lambdify
        # Build model (fresh each time)
        model = self._builder.build()

        # Delegate to model
        return model._lambdify(values, data_for_intermediates)

    def to_flows(self, values=None, flow_ids=None):
        """Return flows data frame with variables substituted by `values`.

        When `flow_ids` is True, assign hash-based flow ids to each row.
        """
        # Extract recipe from values
        recipe_data, other_values = self._extract_recipe_from_values(values)

        # Build model with recipe (fresh each time)
        model = self._builder.build(recipe_data)

        # Delegate to model
        return model.to_flows(other_values, flow_ids)


def convert_indexed_symbols(data):
    """Convert {S[1, 2]: 7} to {S: {(1, 2): 7}}

    This works better with lambdified functions."""
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
