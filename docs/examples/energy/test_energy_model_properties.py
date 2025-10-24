"""
Property-based tests for the energy model example.

Run with: pytest test_energy_model_properties.py -v
For more examples: pytest test_energy_model_properties.py -v --hypothesis-profile=thorough
"""

import pytest
from hypothesis import given, strategies as st, settings, assume, HealthCheck
import pandas as pd
import sys
import sympy as sy

# Load the model
from load_model import model, recipe_data, solution_to_flows


# =============================================================================
# Helper Functions
# =============================================================================

def generate_params(data):
    """Generate random but valid parameter values for the energy model."""
    return {
        # Demand parameters - strictly positive to avoid edge cases
        "Z_1": data.draw(st.floats(min_value=0.1, max_value=100)),  # Steel demand
        "Z_2": data.draw(st.floats(min_value=0.1, max_value=100)),  # Transport demand

        # Capacity limits - non-negative, can be zero
        "S_1": data.draw(st.floats(min_value=0, max_value=50)),     # Wind capacity

        # Allocation fractions - must be in [0, 1]
        "a_1": data.draw(st.floats(min_value=0, max_value=1)),      # Steel production split
    }


def check_object_mass_balance(flows: pd.DataFrame, object_id: str, tolerance=1e-6):
    """
    Check that production equals consumption for a specific object.

    Returns: (is_balanced, production, consumption, difference)
    """
    obj_flows = flows[flows['material'] == object_id]

    # Production: flows where this object appears as target
    production = obj_flows[obj_flows['target'] == object_id]['value'].sum()

    # Consumption: flows where this object appears as source
    consumption = obj_flows[obj_flows['source'] == object_id]['value'].sum()

    difference = abs(production - consumption)
    is_balanced = difference <= tolerance

    return is_balanced, production, consumption, difference


# =============================================================================
# Property Tests
# =============================================================================

@settings(max_examples=100, suppress_health_check=[HealthCheck.filter_too_much])
@given(st.data())
def test_electricity_mass_balance(data):
    """
    Electricity has has_market=True, so production should equal consumption
    for all valid parameter combinations.
    """
    params = generate_params(data)

    try:
        flows = solution_to_flows(model, params)
    except Exception as e:
        pytest.fail(f"Failed to evaluate model with params {params}: {e}")

    is_balanced, prod, cons, diff = check_object_mass_balance(flows, "Electricity")

    assert is_balanced, (
        f"Electricity mass balance violated with params {params}\n"
        f"  Production:  {prod:.6f}\n"
        f"  Consumption: {cons:.6f}\n"
        f"  Difference:  {diff:.6e}"
    )


@settings(max_examples=100, suppress_health_check=[HealthCheck.filter_too_much])
@given(st.data())
def test_hydrogen_mass_balance(data):
    """
    Hydrogen has has_market=True, so production should equal consumption.
    """
    params = generate_params(data)
    flows = solution_to_flows(model, params)

    is_balanced, prod, cons, diff = check_object_mass_balance(flows, "Hydrogen")

    assert is_balanced, (
        f"Hydrogen mass balance violated with params {params}\n"
        f"  Production:  {prod:.6f}\n"
        f"  Consumption: {cons:.6f}\n"
        f"  Difference:  {diff:.6e}"
    )


@settings(max_examples=100)
@given(st.data())
def test_all_market_objects_balance(data):
    """
    All objects with has_market=True should have balanced production and consumption.
    """
    params = generate_params(data)
    flows = solution_to_flows(model, params)

    failures = []
    for obj in model.objects:
        if obj.has_market:
            is_balanced, prod, cons, diff = check_object_mass_balance(flows, obj.id)
            if not is_balanced:
                failures.append(
                    f"{obj.id}: prod={prod:.6f}, cons={cons:.6f}, diff={diff:.6e}"
                )

    assert not failures, (
        f"Mass balance violations with params {params}:\n" +
        "\n".join(failures)
    )


@settings(max_examples=100)
@given(st.data())
def test_all_flows_non_negative(data):
    """All flow values should be non-negative (>= 0)."""
    params = generate_params(data)
    flows = solution_to_flows(model, params)

    # Allow small numerical errors
    negative_flows = flows[flows['value'] < -1e-9]

    assert len(negative_flows) == 0, (
        f"Found {len(negative_flows)} negative flows with params {params}:\n"
        f"{negative_flows[['source', 'target', 'material', 'value']]}"
    )


@settings(max_examples=100)
@given(st.data())
def test_steel_production_allocation(data):
    """
    Steel production should be correctly split between EAF and H2DRI
    according to allocation fraction a_1.
    """
    params = generate_params(data)
    flows = solution_to_flows(model, params)

    Z_1 = params["Z_1"]
    a_1 = params["a_1"]

    # Total steel production should match demand
    total_steel = flows[flows['material'] == 'Steel']['value'].sum()
    assert abs(total_steel - Z_1) < 1e-6, \
        f"Total steel {total_steel} != demand {Z_1}"

    # EAF production
    eaf_steel = flows[
        (flows['source'] == 'SteelProductionEAF') &
        (flows['target'] == 'Steel')
    ]['value'].sum()

    # H2DRI production
    h2dri_steel = flows[
        (flows['source'] == 'SteelProductionH2DRI') &
        (flows['target'] == 'Steel')
    ]['value'].sum()

    # Check allocation
    expected_eaf = Z_1 * a_1
    expected_h2dri = Z_1 * (1 - a_1)

    assert abs(eaf_steel - expected_eaf) < 1e-6, \
        f"EAF steel {eaf_steel} != expected {expected_eaf} (a_1={a_1})"

    assert abs(h2dri_steel - expected_h2dri) < 1e-6, \
        f"H2DRI steel {h2dri_steel} != expected {expected_h2dri} (a_1={1-a_1})"


@settings(max_examples=100)
@given(st.data())
def test_wind_capacity_limit_respected(data):
    """Wind turbine output should never exceed capacity S_1."""
    params = generate_params(data)
    flows = solution_to_flows(model, params)

    wind_output = flows[
        (flows['source'] == 'WindTurbine') &
        (flows['material'] == 'Electricity')
    ]['value'].sum()

    capacity = params['S_1']

    # Allow small numerical tolerance
    assert wind_output <= capacity + 1e-6, \
        f"Wind output {wind_output} exceeded capacity {capacity} with params {params}"


@settings(max_examples=50)
@given(st.data())
def test_wind_capacity_used_when_demand_high(data):
    """
    When electricity demand exceeds wind capacity, wind should produce
    at full capacity.
    """
    # Create scenario with high demand and low wind capacity
    S_1 = data.draw(st.floats(min_value=0.1, max_value=10))   # Low wind capacity
    Z_1 = data.draw(st.floats(min_value=50, max_value=100))   # High steel demand
    Z_2 = data.draw(st.floats(min_value=50, max_value=100))   # High transport demand
    a_1 = data.draw(st.floats(min_value=0, max_value=1))

    params = {"S_1": S_1, "Z_1": Z_1, "Z_2": Z_2, "a_1": a_1}
    flows = solution_to_flows(model, params)

    # Calculate total electricity demand
    total_elec_consumption = flows[flows['material'] == 'Electricity']['value'].sum() / 2  # Divide by 2 since each flow appears twice

    # Get wind output
    wind_output = flows[
        (flows['source'] == 'WindTurbine') &
        (flows['material'] == 'Electricity')
    ]['value'].sum()

    # If demand >> capacity, wind should be at capacity
    # (allowing some tolerance for edge cases)
    if total_elec_consumption > S_1 * 1.5:
        assert abs(wind_output - S_1) < 1e-6, \
            f"Wind should be at capacity {S_1}, got {wind_output} (demand={total_elec_consumption})"


@settings(max_examples=50)
@given(st.data())
def test_wind_capacity_unused_when_demand_low(data):
    """
    When electricity demand is less than wind capacity, wind should only
    produce what's needed (not waste capacity).
    """
    # Create scenario with low demand and high wind capacity
    S_1 = data.draw(st.floats(min_value=50, max_value=100))  # High capacity
    Z_1 = data.draw(st.floats(min_value=0.1, max_value=5))   # Low steel demand
    Z_2 = data.draw(st.floats(min_value=0.1, max_value=5))   # Low transport demand
    a_1 = data.draw(st.floats(min_value=0, max_value=1))

    params = {"S_1": S_1, "Z_1": Z_1, "Z_2": Z_2, "a_1": a_1}
    flows = solution_to_flows(model, params)

    # Calculate electricity consumption
    elec_consumption = flows[
        (flows['source'] == 'Electricity')
    ]['value'].sum()

    # Get wind output
    wind_output = flows[
        (flows['source'] == 'WindTurbine') &
        (flows['material'] == 'Electricity')
    ]['value'].sum()

    # Wind should not overproduce
    assert wind_output <= elec_consumption + 1e-6, \
        f"Wind output {wind_output} > consumption {elec_consumption}"


@settings(max_examples=100)
@given(st.data())
def test_process_no_stock_consistency(data):
    """
    For processes without stock accumulation, input magnitude (X)
    should equal output magnitude (Y).
    """
    params = generate_params(data)

    failures = []
    for j, process in enumerate(model.processes):
        if not process.has_stock:
            # Evaluate X and Y for this process
            try:
                x_val = float(model.eval(model.X[j]).subs({**recipe_data, **params}))
                y_val = float(model.eval(model.Y[j]).subs({**recipe_data, **params}))
            except Exception as e:
                failures.append(f"{process.id}: evaluation failed: {e}")
                continue

            if abs(x_val - y_val) > 1e-6:
                failures.append(
                    f"{process.id}: X={x_val:.6f} != Y={y_val:.6f}"
                )

    assert not failures, (
        f"Process X!=Y violations with params {params}:\n" +
        "\n".join(failures)
    )


@settings(max_examples=100)
@given(st.data())
def test_ccgt_recipe_consistency(data):
    """
    CCGT should consume natural gas according to recipe coefficients.
    Gas consumption = electricity production * (U_gas / S_electricity)
    """
    params = generate_params(data)
    flows = solution_to_flows(model, params)

    # CCGT electricity output
    ccgt_elec = flows[
        (flows['source'] == 'CCGT') &
        (flows['target'] == 'Electricity')
    ]['value'].sum()

    # CCGT gas input
    ccgt_gas = flows[
        (flows['source'] == 'NaturalGas') &
        (flows['target'] == 'CCGT')
    ]['value'].sum()

    # Get recipe coefficients
    i_gas = model._lookup_object('NaturalGas')
    i_elec = model._lookup_object('Electricity')
    j_ccgt = model._lookup_process('CCGT')

    u_gas = recipe_data[model.U[i_gas, j_ccgt]]
    s_elec = recipe_data[model.S[i_elec, j_ccgt]]

    # Calculate expected gas consumption
    expected_gas = ccgt_elec * (u_gas / s_elec)

    # Only check if CCGT is active (producing > 0)
    if ccgt_elec > 1e-6:
        assert abs(ccgt_gas - expected_gas) < 1e-6, (
            f"CCGT gas consumption {ccgt_gas} != expected {expected_gas}\n"
            f"  Electricity output: {ccgt_elec}\n"
            f"  Recipe ratio: {u_gas}/{s_elec} = {u_gas/s_elec}"
        )


@settings(max_examples=100)
@given(st.data())
def test_transport_demand_satisfied(data):
    """Transport service demand Z_2 should be exactly satisfied."""
    params = generate_params(data)
    flows = solution_to_flows(model, params)

    transport_production = flows[
        (flows['target'] == 'TransportService')
    ]['value'].sum()

    expected_demand = params['Z_2']

    assert abs(transport_production - expected_demand) < 1e-6, \
        f"Transport production {transport_production} != demand {expected_demand}"


@settings(max_examples=100)
@given(st.data())
def test_steel_demand_satisfied(data):
    """Steel demand Z_1 should be exactly satisfied."""
    params = generate_params(data)
    flows = solution_to_flows(model, params)

    steel_production = flows[
        (flows['target'] == 'Steel')
    ]['value'].sum()

    expected_demand = params['Z_1']

    assert abs(steel_production - expected_demand) < 1e-6, \
        f"Steel production {steel_production} != demand {expected_demand}"


# =============================================================================
# Edge Case Tests
# =============================================================================

@given(st.data())
def test_zero_demand(data):
    """Model should handle zero demand gracefully."""
    params = {
        "Z_1": 0.0,
        "Z_2": 0.0,
        "S_1": data.draw(st.floats(min_value=0, max_value=50)),
        "a_1": data.draw(st.floats(min_value=0, max_value=1)),
    }

    flows = solution_to_flows(model, params)

    # All flows should be zero (or very close)
    assert (flows['value'] < 1e-6).all(), \
        f"Expected all zero flows with zero demand, got:\n{flows[flows['value'] > 1e-6]}"


@given(st.data())
def test_zero_wind_capacity(data):
    """Model should handle zero wind capacity (CCGT does everything)."""
    params = {
        "Z_1": data.draw(st.floats(min_value=0.1, max_value=100)),
        "Z_2": data.draw(st.floats(min_value=0.1, max_value=100)),
        "S_1": 0.0,  # No wind
        "a_1": data.draw(st.floats(min_value=0, max_value=1)),
    }

    flows = solution_to_flows(model, params)

    wind_output = flows[
        (flows['source'] == 'WindTurbine') &
        (flows['material'] == 'Electricity')
    ]['value'].sum()

    assert wind_output < 1e-6, \
        f"Wind output should be zero when capacity is zero, got {wind_output}"

    # CCGT should provide all electricity
    ccgt_output = flows[
        (flows['source'] == 'CCGT') &
        (flows['material'] == 'Electricity')
    ]['value'].sum()

    elec_consumption = flows[
        (flows['source'] == 'Electricity')
    ]['value'].sum()

    assert abs(ccgt_output - elec_consumption) < 1e-6


@given(st.data())
def test_extreme_allocation(data):
    """Test extreme allocation values (0 and 1)."""
    for a_1 in [0.0, 1.0]:
        params = {
            "Z_1": data.draw(st.floats(min_value=0.1, max_value=100)),
            "Z_2": data.draw(st.floats(min_value=0.1, max_value=100)),
            "S_1": data.draw(st.floats(min_value=0, max_value=50)),
            "a_1": a_1,
        }

        flows = solution_to_flows(model, params)

        # Check that flows are still valid
        is_balanced, _, _, _ = check_object_mass_balance(flows, "Electricity")
        assert is_balanced

        assert (flows['value'] >= -1e-9).all()


# =============================================================================
# Run Configuration
# =============================================================================

if __name__ == "__main__":
    # Run with more verbose output
    pytest.main([__file__, "-v", "--tb=short"])
