"""Acceptance test 4: conservation on the petrochemical baseline.

Runs flowprog.allocation.Allocation on the full petrochem model (143
processes, 111 objects) at a real reference scenario's parameter values, and
checks the 100% conservation rule holds -- both whole-system and
cradle-to-gate (excluding the end-of-life process group, which exercises the
has_stock net-accumulation "stock term" via the UseOf* processes).
"""

import pytest

from structure import load_data, build_structure
from model import define_model
from model_polymers import PROCESS_GROUPS
from flowprog.allocation import Allocation, MassAllocation, Scope


def _build_evaluable_model():
    data = load_data()
    model_builder, recipe_data = build_structure(data)
    define_model(
        model_builder,
        recipe_data,
        data["processes_with_process_emissions"],
    )
    model = model_builder.build(recipe_data)
    scenario = next(iter(data["scenarios"].values()))
    return model, scenario["params"]


@pytest.fixture(scope="module")
def evaluable_model():
    return _build_evaluable_model()


def test_conservation_whole_system(evaluable_model):
    model, params = evaluable_model
    result = Allocation(model, params, MassAllocation()).result
    assert result.check_conservation(atol=1e-4)


def test_conservation_cradle_to_gate_with_stock_terms(evaluable_model):
    """Excluding end-of-life exercises has_stock (X != Y) UseOf* processes,
    whose net accumulation shows up as a conservation-check stock term."""
    model, params = evaluable_model
    scope = Scope(excluded_processes=frozenset(PROCESS_GROUPS["end_of_life"]))
    result = Allocation(model, params, MassAllocation(), scope=scope).result

    has_stock_in_scope = [
        p.id
        for p in model.processes
        if p.has_stock and p.id not in scope.excluded_processes
    ]
    assert has_stock_in_scope  # sanity check the UseOf* processes are in scope

    assert result.check_conservation(atol=1e-4)
