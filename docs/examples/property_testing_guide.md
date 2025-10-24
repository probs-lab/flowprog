# Property-Based Testing Guide for Flowprog Models

This guide shows how to use Hypothesis to verify that your flowprog model maintains important invariants (like mass balance) across different parameter values.

## Why Property-Based Testing for Your Model?

Your flowprog model creates **symbolic SymPy expressions** that are correct algebraically, but when evaluated with **specific numeric parameters**, bugs can emerge:

- Mass balance might break if allocation coefficients don't sum to 1
- Capacity limits might be violated due to floating point errors
- Negative flows might appear if parameters are invalid
- Objects with `has_market=True` might not balance supply and demand

Property-based testing generates hundreds of random parameter combinations to find edge cases.

## Installation

```bash
pip install hypothesis
```

## Basic Pattern

```python
from hypothesis import given, strategies as st
import pytest

@given(st.data())
def test_my_model_properties(data):
    # 1. Generate random parameter values
    params = generate_random_params(data)

    # 2. Evaluate your model with those params
    flows = model.to_flows({**recipe_data, **params})

    # 3. Check invariants
    assert check_mass_balance(flows)
    assert check_non_negativity(flows)
```

## Property 1: Mass Balance for Market Objects

**What to test**: Objects with `has_market=True` should have equal total production and consumption.

**The Issue**: Your symbolic model might use `object_production_deficit()` correctly, but specific parameter values could break the balance.

```python
from hypothesis import given, strategies as st
import pandas as pd

def check_object_mass_balance(flows: pd.DataFrame, object_id: str, tolerance=1e-9):
    """Check that production equals consumption for a specific object."""
    obj_flows = flows[flows['material'] == object_id]

    # Production: flows where this object is produced (target)
    production = obj_flows[obj_flows['target'] == object_id]['value'].sum()

    # Consumption: flows where this object is consumed (source)
    consumption = obj_flows[obj_flows['source'] == object_id]['value'].sum()

    return abs(production - consumption) <= tolerance

@given(st.data())
def test_electricity_mass_balance(data):
    """Electricity should balance across all parameter combinations."""
    # Generate random but valid parameter values
    Z_1 = data.draw(st.floats(min_value=0, max_value=100))  # Steel demand
    Z_2 = data.draw(st.floats(min_value=0, max_value=100))  # Transport demand
    S_1 = data.draw(st.floats(min_value=0, max_value=50))   # Wind capacity
    a_1 = data.draw(st.floats(min_value=0, max_value=1))    # Allocation fraction

    params = {"Z_1": Z_1, "Z_2": Z_2, "S_1": S_1, "a_1": a_1}
    flows = solution_to_flows(model, params)

    # Electricity has has_market=True, so it must balance
    assert check_object_mass_balance(flows, "Electricity"), \
        f"Electricity imbalance with params {params}"
```

**Advanced version** - test all market objects:

```python
@given(st.data())
def test_all_market_objects_balance(data):
    """All objects with has_market=True should balance."""
    params = generate_params(data)
    flows = solution_to_flows(model, params)

    for obj in model.objects:
        if obj.has_market:
            assert check_object_mass_balance(flows, obj.id), \
                f"{obj.id} imbalance: {get_imbalance_details(flows, obj.id)}"

def get_imbalance_details(flows, object_id):
    """Helper to debug mass balance failures."""
    obj_flows = flows[flows['material'] == object_id]
    prod = obj_flows[obj_flows['target'] == object_id]['value'].sum()
    cons = obj_flows[obj_flows['source'] == object_id]['value'].sum()
    return f"production={prod:.6f}, consumption={cons:.6f}, diff={abs(prod-cons):.6e}"
```

## Property 2: Non-Negativity of All Flows

**What to test**: All flow values should be ≥ 0 (no negative production or consumption).

**The Issue**: Invalid parameter values (negative demands, allocations > 1) can create negative flows.

```python
@given(st.data())
def test_all_flows_non_negative(data):
    """All flows should be non-negative for valid parameters."""
    params = generate_params(data)
    flows = solution_to_flows(model, params)

    negative_flows = flows[flows['value'] < -1e-9]  # Small tolerance for numerical error

    assert len(negative_flows) == 0, \
        f"Found {len(negative_flows)} negative flows:\n{negative_flows}"
```

## Property 3: Allocation Coefficients Sum to One

**What to test**: When you use allocation fractions (like `a_1` for steel production), related flows should sum correctly.

**The Issue**: If allocation parameters don't sum to 1, mass balance breaks.

```python
@given(st.data())
def test_steel_production_allocation(data):
    """Steel production should split correctly between EAF and H2DRI."""
    Z_1 = data.draw(st.floats(min_value=0.1, max_value=100))
    a_1 = data.draw(st.floats(min_value=0, max_value=1))

    # Other params with safe defaults
    params = {"Z_1": Z_1, "Z_2": 1.0, "S_1": 10.0, "a_1": a_1}
    flows = solution_to_flows(model, params)

    # Total steel production
    total_steel = flows[flows['material'] == 'Steel']['value'].sum()

    # Individual process outputs
    eaf_steel = flows[
        (flows['source'] == 'SteelProductionEAF') &
        (flows['target'] == 'Steel')
    ]['value'].sum()

    h2dri_steel = flows[
        (flows['source'] == 'SteelProductionH2DRI') &
        (flows['target'] == 'Steel')
    ]['value'].sum()

    # Check allocation works correctly
    assert abs(total_steel - Z_1) < 1e-6, "Total steel should match demand"
    assert abs(eaf_steel - Z_1 * a_1) < 1e-6, "EAF steel should be a_1 fraction"
    assert abs(h2dri_steel - Z_1 * (1 - a_1)) < 1e-6, "H2DRI should be (1-a_1) fraction"
```

## Property 4: Capacity Limits Are Respected

**What to test**: When using `model.limit()`, the limited value should never exceed the limit.

**The Issue**: Piecewise expressions can be tricky - you might have logic errors in limit application.

```python
@given(st.data())
def test_wind_capacity_limit_respected(data):
    """Wind turbine output should never exceed S_1 capacity."""
    params = generate_params(data)
    flows = solution_to_flows(model, params)

    wind_output = flows[
        (flows['source'] == 'WindTurbine') &
        (flows['material'] == 'Electricity')
    ]['value'].sum()

    capacity = params['S_1']

    assert wind_output <= capacity + 1e-6, \
        f"Wind output {wind_output} exceeded capacity {capacity}"

@given(st.data())
def test_capacity_limit_used_when_possible(data):
    """Wind should produce at capacity when demand exceeds capacity."""
    # Set up scenario where demand > capacity
    S_1 = data.draw(st.floats(min_value=1, max_value=10))  # Small capacity
    Z_1 = data.draw(st.floats(min_value=100, max_value=200))  # Large steel demand
    Z_2 = data.draw(st.floats(min_value=100, max_value=200))  # Large transport demand

    params = {"S_1": S_1, "Z_1": Z_1, "Z_2": Z_2, "a_1": 0.5}
    flows = solution_to_flows(model, params)

    wind_output = flows[
        (flows['source'] == 'WindTurbine') &
        (flows['material'] == 'Electricity')
    ]['value'].sum()

    # When demand is high, wind should produce at full capacity
    assert abs(wind_output - S_1) < 1e-6, \
        f"Wind should be at capacity {S_1} when demand is high, got {wind_output}"
```

## Property 5: Process Input-Output Consistency

**What to test**: For processes without stock, `X[j] == Y[j]` (input magnitude equals output magnitude).

**The Issue**: Recipe coefficients might create inconsistencies.

```python
@given(st.data())
def test_process_no_stock_balance(data):
    """Processes without stock should have X_j == Y_j."""
    params = generate_params(data)

    # Evaluate X and Y for each process
    for j, process in enumerate(model.processes):
        if not process.has_stock:
            x_val = model.eval(model.X[j]).subs({**recipe_data, **params})
            y_val = model.eval(model.Y[j]).subs({**recipe_data, **params})

            assert abs(float(x_val) - float(y_val)) < 1e-6, \
                f"Process {process.id}: X={x_val} != Y={y_val} with params {params}"
```

## Property 6: Recipe Consistency

**What to test**: Process flows should respect recipe coefficients (U, S matrices).

```python
@given(st.data())
def test_recipe_coefficients_respected(data):
    """Process input/output flows should match recipe (U, S) coefficients."""
    params = generate_params(data)
    flows = solution_to_flows(model, params)

    # For CCGT: should consume natural gas according to U coefficient
    ccgt_elec_output = flows[
        (flows['source'] == 'CCGT') &
        (flows['target'] == 'Electricity')
    ]['value'].sum()

    ccgt_gas_input = flows[
        (flows['source'] == 'NaturalGas') &
        (flows['target'] == 'CCGT')
    ]['value'].sum()

    # Calculate expected ratio from recipe
    i_gas = model._lookup_object('NaturalGas')
    i_elec = model._lookup_object('Electricity')
    j_ccgt = model._lookup_process('CCGT')

    expected_ratio = recipe_data[model.U[i_gas, j_ccgt]] / recipe_data[model.S[i_elec, j_ccgt]]

    if ccgt_elec_output > 1e-6:  # Only check if process is active
        actual_ratio = ccgt_gas_input / ccgt_elec_output
        assert abs(actual_ratio - expected_ratio) < 1e-6, \
            f"CCGT gas/electricity ratio {actual_ratio} != expected {expected_ratio}"
```

## Helper Function: Generate Valid Parameters

Create a strategy that generates realistic parameter values:

```python
import hypothesis.strategies as st

def generate_params(data):
    """Generate a valid random parameter set for the energy model."""
    return {
        # Demand parameters - non-negative
        "Z_1": data.draw(st.floats(min_value=0, max_value=100)),
        "Z_2": data.draw(st.floats(min_value=0, max_value=100)),

        # Capacity limits - non-negative
        "S_1": data.draw(st.floats(min_value=0, max_value=50)),

        # Allocation fractions - must be in [0, 1]
        "a_1": data.draw(st.floats(min_value=0, max_value=1)),
    }

# If you have multiple allocation parameters that must sum to 1:
def generate_allocation_params(data, n_options):
    """Generate n allocation fractions that sum to 1."""
    # Generate n-1 random values
    values = [data.draw(st.floats(min_value=0, max_value=1)) for _ in range(n_options - 1)]
    values = sorted(values)

    # Convert to fractions that sum to 1
    fractions = []
    prev = 0
    for v in values:
        fractions.append(v - prev)
        prev = v
    fractions.append(1 - prev)

    return fractions
```

## Running Your Tests

```python
# In your test file (e.g., test_energy_model.py)
import pytest
from hypothesis import given, settings, strategies as st
from load_model import model, recipe_data, solution_to_flows

# Run tests with more examples for thorough checking
@settings(max_examples=200)
@given(st.data())
def test_model_mass_balance_comprehensive(data):
    params = generate_params(data)
    flows = solution_to_flows(model, params)

    # Check all invariants
    for obj in model.objects:
        if obj.has_market:
            assert check_object_mass_balance(flows, obj.id)

    assert (flows['value'] >= -1e-9).all()
```

Run with pytest:
```bash
pytest test_energy_model.py -v
# For faster feedback during development:
pytest test_energy_model.py -v --hypothesis-seed=0 --maxfail=1
```

## Finding Bugs with Hypothesis

When Hypothesis finds a failure, it **shrinks** to the minimal example:

```
Falsifying example: test_electricity_mass_balance(
    data=data(...),
)
Draw 1: Z_1=0.0
Draw 2: Z_2=0.0
Draw 3: S_1=0.0
Draw 4: a_1=0.5

AssertionError: Electricity imbalance: production=0.0, consumption=-0.001
```

This tells you exactly which parameter combination breaks your model!

## Advanced: Testing Symbolic Expressions

You can also test properties of the **symbolic expressions** before evaluation:

```python
import sympy as sy

def test_symbolic_mass_balance():
    """Check that electricity production - consumption is symbolically zero or non-negative."""
    elec_production = model.object_production("Electricity")
    elec_consumption = model.object_consumption("Electricity")

    balance = sy.simplify(elec_production - elec_consumption)

    # The balance should simplify to zero (perfect balance)
    # or to a Max(0, ...) expression (deficit handled by CCGT)
    assert balance == 0 or isinstance(balance, sy.Max)
```

## Common Patterns for Different Model Types

### Linear Supply Chain
```python
# Objects: input -> intermediate -> output
# Property: output flow * recipe = input flow

@given(st.data())
def test_chain_consistency(data):
    demand = data.draw(st.floats(min_value=0, max_value=100))
    flows = model.to_flows({"demand": demand, **recipe_data})

    output_flow = flows[flows['target'] == 'output']['value'].sum()
    input_flow = flows[flows['source'] == 'input']['value'].sum()

    expected_ratio = calculate_chain_ratio(model, recipe_data)
    assert abs(output_flow * expected_ratio - input_flow) < 1e-6
```

### Multi-Producer with Allocation
```python
@given(st.data())
def test_merit_order_dispatch(data):
    """When using merit order (allocate to cheapest first), verify dispatch order."""
    demand = data.draw(st.floats(min_value=0, max_value=100))
    capacity_cheap = data.draw(st.floats(min_value=0, max_value=50))

    flows = model.to_flows({"demand": demand, "cap_cheap": capacity_cheap})

    cheap_output = flows[flows['source'] == 'cheap_producer']['value'].sum()
    expensive_output = flows[flows['source'] == 'expensive_producer']['value'].sum()

    # Cheap producer should max out before expensive is used
    if demand <= capacity_cheap:
        assert abs(cheap_output - demand) < 1e-6
        assert expensive_output < 1e-6
    else:
        assert abs(cheap_output - capacity_cheap) < 1e-6
        assert abs(expensive_output - (demand - capacity_cheap)) < 1e-6
```

### Recycling Loop
```python
@given(st.data())
def test_recycling_balance(data):
    """Material in recycling loop should balance: waste collected = material recycled + losses."""
    params = generate_params(data)
    flows = model.to_flows({**recipe_data, **params})

    waste_collected = flows[flows['target'] == 'recycling_process']['value'].sum()
    recycled_material = flows[flows['source'] == 'recycling_process']['value'].sum()

    # Accounting for losses
    loss_fraction = recipe_data.get("recycling_loss", 0)
    expected_recycled = waste_collected * (1 - loss_fraction)

    assert abs(recycled_material - expected_recycled) < 1e-6
```

## Summary

Property-based testing for your flowprog model should verify:

1. ✅ **Mass balance** for objects with `has_market=True`
2. ✅ **Non-negativity** of all flows
3. ✅ **Allocation coefficients** sum correctly
4. ✅ **Capacity limits** are respected
5. ✅ **Process consistency** (X=Y for no-stock processes)
6. ✅ **Recipe coefficients** are applied correctly

The key insight: Your **symbolic model** is correct, but **numeric evaluation** can reveal edge cases. Property-based testing finds them automatically!
