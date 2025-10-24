# Debugging Property Test Failures in Flowprog Models

When Hypothesis finds a failing test case, it will shrink it to the minimal failing example. This guide shows how to interpret and debug those failures.

## Example Failure: Mass Balance Violation

### What Hypothesis Reports

```
FAILED test_electricity_mass_balance
Falsifying example: test_electricity_mass_balance(
    data=data(...)
)
Draw 1 (Z_1): 10.0
Draw 2 (Z_2): 5.0
Draw 3 (S_1): 0.0
Draw 4 (a_1): 0.5

AssertionError: Electricity mass balance violated with params
{'Z_1': 10.0, 'Z_2': 5.0, 'S_1': 0.0, 'a_1': 0.5}
  Production:  50.500000
  Consumption: 50.000000
  Difference:  5.000000e-01
```

### How to Debug

#### Step 1: Reproduce the Exact Failure

```python
# In a notebook or interactive Python session
from load_model import model, recipe_data, solution_to_flows

# Use the exact parameters from the failure
params = {'Z_1': 10.0, 'Z_2': 5.0, 'S_1': 0.0, 'a_1': 0.5}
flows = solution_to_flows(model, params)

# Look at all flows
print(flows)
```

#### Step 2: Inspect Electricity Flows Specifically

```python
# Filter to electricity flows only
elec_flows = flows[flows['material'] == 'Electricity']
print("\nElectricity flows:")
print(elec_flows[['source', 'target', 'value']])

# Calculate production and consumption
production = elec_flows[elec_flows['target'] == 'Electricity']['value'].sum()
consumption = elec_flows[elec_flows['source'] == 'Electricity']['value'].sum()

print(f"\nProduction:  {production}")
print(f"Consumption: {consumption}")
print(f"Difference:  {abs(production - consumption)}")
```

#### Step 3: Check Symbolic Expressions

```python
# Look at the symbolic expressions before substitution
print("\nWind turbine output (symbolic):")
j_wind = model._lookup_process('WindTurbine')
print(model[model.Y[j_wind]])

print("\nCCGT output (symbolic):")
j_ccgt = model._lookup_process('CCGT')
print(model[model.Y[j_ccgt]])

# Evaluate with the failing parameters
wind_output = model.eval(model.Y[j_wind]).subs({**recipe_data, **params})
ccgt_output = model.eval(model.Y[j_ccgt]).subs({**recipe_data, **params})

print(f"\nWind output (numeric): {wind_output}")
print(f"CCGT output (numeric): {ccgt_output}")
```

#### Step 4: Trace the Model Building Steps

```python
# Look at the history of how expressions were built
print("\nModel building history for CCGT output:")
print("\n".join(model.get_history(model.Y[j_ccgt])))

# Check the production deficit calculation
i_elec = model._lookup_object('Electricity')
deficit = model.object_production_deficit('Electricity')
print(f"\nElectricity production deficit (symbolic):")
print(deficit)

deficit_value = model.eval(deficit).subs({**recipe_data, **params})
print(f"Electricity production deficit (numeric): {deficit_value}")
```

#### Step 5: Identify the Root Cause

Common causes of mass balance failures:

1. **Piecewise expressions with edge cases**
   ```python
   # Check if limit() is causing issues
   # Look for Max(0, ...) or Piecewise(...) in expressions
   ```

2. **Floating point precision**
   ```python
   # Check if difference is very small
   if abs(production - consumption) < 1e-6:
       print("This is just floating point error, adjust tolerance")
   ```

3. **Missing `until_objects`**
   ```python
   # If you pull production but don't stop propagation,
   # you might be counting flows twice
   # Solution: use until_objects parameter
   ```

4. **Incorrect allocation coefficients**
   ```python
   # Check if allocations sum to 1
   # For the failing example, check a_1 + (1-a_1) = 1
   ```

## Example Failure: Negative Flows

### What Hypothesis Reports

```
FAILED test_all_flows_non_negative
Found 2 negative flows with params {'Z_1': 0.0, 'Z_2': 100.0, 'S_1': 10.0, 'a_1': 1.5}
   source  target    material     value
3  Process1  Output   Material  -15.2
```

### How to Debug

```python
params = {'Z_1': 0.0, 'Z_2': 100.0, 'S_1': 10.0, 'a_1': 1.5}

# The problem is obvious: a_1 = 1.5 is invalid!
# Allocation fractions must be in [0, 1]

# Fix: Update your parameter generation strategy
def generate_params(data):
    return {
        "Z_1": data.draw(st.floats(min_value=0, max_value=100)),
        "Z_2": data.draw(st.floats(min_value=0, max_value=100)),
        "S_1": data.draw(st.floats(min_value=0, max_value=50)),
        "a_1": data.draw(st.floats(min_value=0, max_value=1)),  # ← constrain to [0,1]
    }
```

## Example Failure: Capacity Limit Exceeded

### What Hypothesis Reports

```
FAILED test_wind_capacity_limit_respected
Wind output 15.2 exceeded capacity 10.0 with params {...}
```

### How to Debug

```python
# Check the limit() application in your model
print(model[model.Y[j_wind]])

# Look for the Piecewise expression
# It should be: Piecewise((0, demand <= 0), (S_1, S_1 <= demand), (demand, True))

# If it's not there, you forgot to apply the limit
# Or the limit expression is wrong

# Verify the limit expression
from sympy import symbols
S_1, demand = symbols('S_1 demand', positive=True)

# Correct limit behavior:
# - If demand <= 0: produce 0
# - If demand >= S_1: produce S_1 (at capacity)
# - Otherwise: produce demand (partial capacity)
```

## Common Debugging Patterns

### Pattern 1: Visualize the Flow Network

```python
import networkx as nx
import matplotlib.pyplot as plt

def flows_to_graph(flows, threshold=1e-6):
    """Convert flows DataFrame to NetworkX graph for visualization."""
    G = nx.DiGraph()

    for _, row in flows.iterrows():
        if row['value'] > threshold:
            G.add_edge(
                row['source'],
                row['target'],
                weight=row['value'],
                label=f"{row['value']:.2f}"
            )

    return G

G = flows_to_graph(flows)
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=2000, node_color='lightblue')
edge_labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(G, pos, edge_labels)
plt.show()
```

### Pattern 2: Compare Symbolic vs Numeric

```python
def compare_symbolic_numeric(model, symbol, params):
    """Show both symbolic expression and numeric value."""
    symbolic = model[symbol]
    numeric = float(model.eval(symbol).subs({**recipe_data, **params}))

    print(f"Symbolic: {symbolic}")
    print(f"Numeric:  {numeric}")
    return symbolic, numeric

# Example usage
compare_symbolic_numeric(model, model.Y[0], params)
```

### Pattern 3: Check Intermediate Values

```python
def evaluate_all_intermediates(model, params):
    """Evaluate all intermediate expressions in the model."""
    results = {}

    for symbol, expr in model._intermediates.items():
        try:
            value = float(model.eval_intermediates(expr, {**recipe_data, **params}))
            results[str(symbol)] = value
        except Exception as e:
            results[str(symbol)] = f"Error: {e}"

    return pd.DataFrame.from_dict(results, orient='index', columns=['value'])

intermediates = evaluate_all_intermediates(model, params)
print(intermediates.sort_values('value', ascending=False))
```

### Pattern 4: Sensitivity Analysis

```python
# If a test barely fails, check parameter sensitivity
def check_sensitivity(model, base_params, vary_param, values):
    """Check how varying one parameter affects mass balance."""
    results = []

    for val in values:
        params = {**base_params, vary_param: val}
        flows = solution_to_flows(model, params)

        is_bal, prod, cons, diff = check_mass_balance(flows, 'Electricity')

        results.append({
            vary_param: val,
            'production': prod,
            'consumption': cons,
            'difference': diff,
            'balanced': is_bal
        })

    return pd.DataFrame(results)

# Example: vary S_1 around the failing value
base = {'Z_1': 10.0, 'Z_2': 5.0, 'a_1': 0.5}
sensitivity = check_sensitivity(model, base, 'S_1', [0, 1, 5, 10, 20, 50])
print(sensitivity)
```

## Fixing Common Issues

### Issue: "Mass balance is off by a small amount (1e-7)"

**Cause**: Floating point arithmetic

**Fix**: Increase tolerance in assertions
```python
assert abs(production - consumption) < 1e-6  # Instead of exact equality
```

### Issue: "Fails only when capacity is exactly 0"

**Cause**: Edge case in Piecewise expression

**Fix**: Handle zero capacity explicitly
```python
limited = model.limit(
    production_dict,
    expr=model.Y[j],
    limit=sy.Max(capacity, 1e-10)  # Avoid exact zero
)
```

### Issue: "Fails with certain allocation combinations"

**Cause**: Allocations don't sum to 1

**Fix**: Use complementary fractions
```python
allocate_backwards={
    "Steel": {
        "EAF": alpha,
        "H2DRI": 1 - alpha,  # Ensures sum = 1
    }
}
```

### Issue: "Model is correct symbolically but fails numerically"

**Cause**: Parameter constraints violated

**Fix**: Add assumptions or filter in Hypothesis
```python
@given(st.data())
def test_model(data):
    params = generate_params(data)

    # Add runtime constraint
    assume(params['efficiency'] > 0.1)  # Must be reasonable
    assume(params['capacity'] >= params['minimum_output'])

    # Now test...
```

## Using Hypothesis Seed for Reproducibility

When Hypothesis finds a failure, it prints the seed:

```
Falsified on the first call with seed=123456
```

To reproduce exactly:

```bash
pytest test_model.py --hypothesis-seed=123456
```

Or in the test:
```python
@settings(max_examples=100, derandomize=True)  # Always uses seed 0
```

## Summary

When debugging property test failures:

1. ✅ **Reproduce** the exact failure with the reported parameters
2. ✅ **Inspect** the flows DataFrame to see what's wrong
3. ✅ **Trace** symbolic expressions to numeric values
4. ✅ **Identify** root cause (edge case, constraint violation, logic error)
5. ✅ **Fix** either the model or the test (parameter constraints)
6. ✅ **Verify** fix by running the specific seed again

The beauty of property-based testing: Hypothesis finds the edge cases you'd never think to test manually!
