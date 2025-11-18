# Advanced Patterns and Techniques

This guide covers advanced patterns for building complex flow programming models with flowprog.

## Pattern 1: Circular Economy with Recycling Priority

A common scenario in circular economy modeling: prioritize recycled materials before virgin production.

```python
import sympy as sy
from flowprog.imperative_model import Model, Process, Object

# Define processes
processes = [
    Process(id="Recycling", consumes=["Waste"], produces=["RecycledMaterial"]),
    Process(id="VirginProduction", consumes=["RawMaterial"], produces=["VirginMaterial"]),
    Process(id="Manufacturing", consumes=["Material"], produces=["Product"]),
]

objects = [
    Object(id="Waste", metric=..., has_market=False),
    Object(id="RecycledMaterial", metric=..., has_market=True),
    Object(id="VirginMaterial", metric=..., has_market=True),
    Object(id="Material", metric=..., has_market=True),  # Generic material
    Object(id="RawMaterial", metric=..., has_market=False),
    Object(id="Product", metric=..., has_market=False),
]

model = Model(processes, objects)

# Step 1: Define product demand
product_demand = sy.Symbol("demand")
model.add(
    model.pull_production("Product", product_demand, until_objects=["Material"]),
    label="Product demand"
)

# Step 2: Use available recycled material first
recycled_supply = sy.Symbol("recycled_available")
recycled_flows = model.limit(
    model.pull_process_output("Recycling", "RecycledMaterial", recycled_supply),
    expr=model.expr("ProcessOutput", process_id="Recycling", object_id="RecycledMaterial"),
    limit=model.expr("Consumption", object_id="Material")  # Don't exceed demand
)
model.add(recycled_flows, label="Recycled material supply (priority)")

# Step 3: Fill remaining gap with virgin production
material_deficit = model.object_production_deficit("Material")
model.add(
    model.pull_production("VirginMaterial", material_deficit, ...),
    label="Virgin material (backup)"
)
```

**Key insight**: The `limit()` method ensures recycled material never exceeds demand, and `object_production_deficit()` calculates exactly how much virgin material is needed.

## Pattern 2: Co-product Utilization

Industrial processes often produce multiple outputs. One process's byproduct can supply another process's input.

```python
# Example: Petrochemical production with co-products

# Ethylene cracker produces both ethylene and propylene
ethylene_cracker = Process(
    id="SteamCracker",
    consumes=["Naphtha"],
    produces=["Ethylene", "Propylene"]  # Co-products
)

# Step 1: Pull ethylene demand
model.add(
    model.pull_production("Ethylene", ethylene_demand, until_objects=["Naphtha"]),
    label="Ethylene demand"
)

# This automatically produces propylene as a co-product
# Check how much propylene we have
propylene_surplus = model.object_balance("Propylene")

# Step 2: If propylene demand exceeds co-product supply, add dedicated production
propylene_deficit = model.object_production_deficit("Propylene")
model.add(
    model.pull_production("Propylene", propylene_deficit, ...),
    label="Dedicated propylene production"
)
```

**Key insight**: Pull/push operations automatically calculate co-product generation based on process recipes. Use balancing methods to determine if additional dedicated production is needed.

## Pattern 3: Time-Dynamic Stocks

Model systems where stock levels change over time periods.

```python
# Products in use as a stock
in_use = Process(
    id="ProductsInUse",
    consumes=["NewProducts"],
    produces=["EndOfLifeProducts"],
    has_stock=True
)

# For each time period
for t in time_periods:
    # Additions to stock
    additions_t = sy.Symbol(f"additions_{t}")
    model.add(
        model.push_process_input("ProductsInUse", "NewProducts", additions_t),
        label=f"Additions in period {t}"
    )

    # Removals from stock (based on lifetime distribution)
    removals_t = calculate_removals(t, previous_additions, lifetime_dist)
    model.add(
        model.pull_process_output("ProductsInUse", "EndOfLifeProducts", removals_t),
        label=f"Removals in period {t}"
    )

    # Stock change: Delta_j = additions_t - removals_t
```

**Key insight**: Use `has_stock=True` to decouple inputs from outputs, allowing stock accumulation or depletion.

## Pattern 4: Multi-Stage Merit Order

Create cascading priorities across multiple production options.

```python
# Example: Electricity generation with renewable priority

# Priority 1: Solar (capacity limited, no fuel cost)
solar_capacity = sy.Symbol("solar_cap")
solar_flows = model.limit(
    model.pull_process_output("Solar", "Electricity", solar_capacity),
    expr=model.expr("ProcessOutput", process_id="Solar", object_id="Electricity"),
    limit=model.expr("Consumption", object_id="Electricity")
)
model.add(solar_flows, label="Solar generation (priority 1)")

# Priority 2: Wind (capacity limited, no fuel cost)
wind_capacity = sy.Symbol("wind_cap")
deficit_after_solar = model.object_production_deficit("Electricity")
wind_flows = model.limit(
    model.pull_process_output("Wind", "Electricity", wind_capacity),
    expr=model.expr("ProcessOutput", process_id="Wind", object_id="Electricity"),
    limit=deficit_after_solar
)
model.add(wind_flows, label="Wind generation (priority 2)")

# Priority 3: Natural gas (capacity limited, moderate cost)
gas_capacity = sy.Symbol("gas_cap")
deficit_after_wind = model.object_production_deficit("Electricity")
gas_flows = model.limit(
    model.pull_process_output("NaturalGas", "Electricity", gas_capacity),
    expr=model.expr("ProcessOutput", process_id="NaturalGas", object_id="Electricity"),
    limit=deficit_after_wind
)
model.add(gas_flows, label="Natural gas generation (priority 3)")

# Priority 4: Coal (unlimited backup, highest cost)
deficit_after_gas = model.object_production_deficit("Electricity")
model.add(
    model.pull_process_output("Coal", "Electricity", deficit_after_gas),
    label="Coal generation (backup)"
)
```

**Key insight**: Each stage calculates the remaining deficit after previous sources. This creates realistic dispatch curves.

## Pattern 5: Regional Markets with Allocation

Model systems where demand is split across multiple regions or producers.

```python
# Define multiple producers for the same object
steel_producers = ["EAF_Recycling", "BlastFurnace", "DRI_Hydrogen"]

# Define market shares (can be symbolic for scenario analysis)
market_shares = {
    "EAF_Recycling": sy.Symbol("alpha_EAF"),
    "BlastFurnace": sy.Symbol("alpha_BF"),
    "DRI_Hydrogen": 1 - sy.Symbol("alpha_EAF") - sy.Symbol("alpha_BF")
}

# Pull production with allocation
model.add(
    model.pull_production(
        "Steel",
        total_steel_demand,
        until_objects=["Electricity", "IronOre", "Hydrogen"],
        allocate_backwards={"Steel": market_shares}
    ),
    label="Steel production by technology mix"
)
```

**Key insight**: Allocation coefficients can be symbolic, allowing scenario analysis of technology transitions.

## Pattern 6: Conditional Process Selection

Use SymPy's conditional expressions for more complex logic.

```python
# Example: Switch production method based on feedstock price

# Define price threshold
price_threshold = sy.Symbol("price_threshold")
current_price = sy.Symbol("current_price")

# Allocate based on price
allocation = sy.Piecewise(
    (1, current_price < price_threshold),  # Use cheap method when price low
    (0, True)                              # Use expensive method when price high
)

market_shares = {
    "CheapMethod": allocation,
    "ExpensiveMethod": 1 - allocation
}

model.add(
    model.pull_production("Product", demand, allocate_backwards={"Product": market_shares}),
    label="Price-responsive production"
)
```

**Key insight**: Combine flowprog's discrete logic with SymPy's symbolic conditionals for complex decision rules.

## Pattern 7: Uncertainty Analysis Workflow

Leverage symbolic equations for robust uncertainty analysis.

```python
# Step 1: Define model with symbolic parameters
model = build_model()  # Returns model with symbolic parameters

# Step 2: Compile model for fast evaluation
flows_func = model.lambdify()

# Step 3: Define parameter distributions (using scipy or similar)
from scipy.stats import uniform, norm, lognorm

param_distributions = {
    sy.Symbol("demand"): norm(loc=1000, scale=100),
    sy.Symbol("recycling_rate"): uniform(loc=0.2, scale=0.3),
    sy.Symbol("efficiency"): lognorm(s=0.1, scale=0.85)
}

# Step 4: Monte Carlo sampling
import numpy as np
n_samples = 10000
results = []

for i in range(n_samples):
    # Sample parameters
    sample_params = {
        param: dist.rvs()
        for param, dist in param_distributions.items()
    }

    # Evaluate model with sampled parameters
    flows = flows_func(sample_params)
    results.append(flows)

# Step 5: Analyze results
results_df = pd.DataFrame(results)
print(results_df.describe())
```

**Key insight**: The symbolic model is defined once, then evaluated many times with different parameters. Mass balance is preserved in every sample.

## Pattern 8: Sensitivity Analysis with SALib

Integrate with SALib for global sensitivity analysis.

```python
from SALib.sample import saltelli
from SALib.analyze import sobol

# Define parameter space
problem = {
    'num_vars': 3,
    'names': ['demand', 'recycling_rate', 'virgin_cost'],
    'bounds': [[800, 1200], [0.2, 0.5], [10, 30]]
}

# Generate samples (Saltelli sampling for Sobol indices)
param_values = saltelli.sample(problem, 1024)

# Evaluate model for each sample
compiled_model = model.lambdify()
outputs = []

for params in param_values:
    param_dict = dict(zip(problem['names'], params))
    flows = compiled_model(param_dict)

    # Extract output of interest (e.g., CO2 emissions)
    emissions = flows["CO2_to_atmosphere"]
    outputs.append(emissions)

# Compute Sobol sensitivity indices
Si = sobol.analyze(problem, np.array(outputs))

print("First-order indices:", Si['S1'])
print("Total-order indices:", Si['ST'])
```

**Key insight**: Flowprog's compiled models integrate seamlessly with SALib for efficient sensitivity analysis.

## Pattern 9: Debugging Complex Models

Use history tracking and intermediate expressions for debugging.

```python
# Enable detailed logging
import logging
logging.getLogger("flowprog.imperative_model").setLevel(logging.DEBUG)

# Build model with descriptive labels
model.add(..., label="Steel production from demand")
model.add(..., label="Recycled steel supply")

# After building, inspect specific variables
process_idx = model._lookup_process("BlastFurnace")
output_history = model.get_history(model.Y[process_idx])
print(f"BlastFurnace output was set by: {output_history}")

# Inspect symbolic expression
print(f"Expression: {model.eval(model.Y[process_idx])}")

# Check balance for an object
balance = model.object_balance("Steel")
print(f"Steel balance: {balance}")
```

**Key insight**: History labels and logging help trace how the model was built, making complex models easier to understand and debug.

## Pattern 10: Integration with Brightway for LCA

Use flowprog for foreground system modeling, Brightway for LCA calculations.

```python
# Step 1: Build flowprog model
model = build_industrial_system()

# Step 2: Evaluate for specific scenario
flows_df = model.to_flows(scenario_params)

# Step 3: Map flows to Brightway activities
import brightway2 as bw

lca_inventory = []
for _, flow in flows_df.iterrows():
    # Find corresponding Brightway activity
    activity = find_brightway_activity(flow['source'], flow['target'])
    amount = flow['value']

    lca_inventory.append((activity, amount))

# Step 4: Calculate impacts
functional_unit = {lca_inventory[0][0]: 1.0}
lca = bw.LCA(functional_unit)
lca.lci()
lca.lcia()

print(f"GWP impact: {lca.score}")
```

**Key insight**: Flowprog handles complex foreground system logic; Brightway provides comprehensive background LCA database and impact assessment.

## Best Practices

1. **Use descriptive labels**: Always provide meaningful labels to `add()` for debugging and documentation
2. **Check balances incrementally**: After each major step, verify balances with `object_balance()`
3. **Document assumptions**: Use comments to explain allocation coefficients and capacity limits
4. **Test with simple values first**: Evaluate with concrete numbers before symbolic analysis
5. **Build incrementally**: Start with simple flows, add complexity step by step
6. **Validate physically**: Ensure mass balance holds, no negative flows, capacities respected
7. **Use symbolic parameters**: Keep actual data values separate from model structure
8. **Leverage history tracking**: Use `get_history()` to understand complex models
9. **Profile performance**: Use `lambdify()` for repeated evaluations in uncertainty analysis
10. **Version control your model building code**: The Python script that builds your model is your model specification
