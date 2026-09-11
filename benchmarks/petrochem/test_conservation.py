"""Acceptance test 4: conservation on the petrochemical model.

Runs flowprog.allocation.AllocatedSystem on the full petrochem model at named
scenarios' parameter values, and checks the 100% conservation rule holds --
both whole-system and cradle-to-gate (excluding the end-of-life process
group, which exercises the has_stock net-accumulation "stock burden" via the
UseOf* processes).
"""

import pytest

from structure import load_data, build_structure
from model import define_model
from model_polymers import PROCESS_GROUPS
from flowprog.allocation import AllocatedSystem, ByValue, Scope


#: Scenarios to check. The conservation rule should hold at every operating
#: point, so test multiple scenarios.
SCENARIOS = ["baseline", "all_last"]

#: The burden in the system is of order 1e11 kg CO2e and the residuals come
#: out around 1e-4, so this is floating-point noise with room to spare.
ATOL = 1e-2


@pytest.fixture(scope="module")
def evaluable_model():
    data = load_data()
    model_builder, recipe_data = build_structure(data)
    define_model(
        model_builder,
        recipe_data,
        data["processes_with_process_emissions"],
    )
    return model_builder.build(recipe_data), data["scenarios"]


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_conservation_whole_system(evaluable_model, scenario):
    model, scenarios = evaluable_model
    params = scenarios[scenario]["params"]
    result = AllocatedSystem(model, params, ByValue())
    assert result.check_conservation(atol=ATOL)


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_conservation_cradle_to_gate_with_stock_burden(evaluable_model, scenario):
    """Excluding end-of-life exercises has_stock (X != Y) UseOf* processes,
    whose accumulation shows up as a stock burden."""
    model, scenarios = evaluable_model
    params = scenarios[scenario]["params"]
    scope = Scope(excluded_processes=frozenset(PROCESS_GROUPS["end_of_life"]))
    result = AllocatedSystem(model, params, ByValue(), scope=scope)

    has_stock_in_scope = [
        p.id
        for p in model.processes
        if p.has_stock and p.id not in scope.excluded_processes
    ]
    assert has_stock_in_scope  # sanity check the UseOf* processes are in scope
    assert len(result.stock_burden) > 0

    assert result.check_conservation(atol=ATOL)
