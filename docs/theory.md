# Theory

The structure of the model is consistent with the [PRObs Ontology](https://probs-lab.github.io/probs-ontology), that is:

- The main elements are **Processes**, which represent some transformation, storage, or transportation activities occuring within the model. A Process has an input of some Object(s) and an output of some Object(s).
- An **Object** is a type of thing, including goods, materials and substances, but also non-material things such as energy and services.

The inputs and outputs of a process (its "recipe") can be written as:

$$ \begin{align}
U_{ij} &= \text{use of object $i$ for a unit operation of process $j$} \\
S_{ij} &= \text{supply of object $i$ from a unit operation of process $j$}
\end{align} $$

If the magnitude of output of process $j$ is $Y_j$, then the actual output of material $i$ is scaled up to be $s_{ij} = S_{ij} Y_j$. Similarly, if the magnitude of input into process $j$ is $X_j$, then the actual input of material $i$ is scaled up to be $u_{ij} = U_{ij} X_j$.

In many processes, no accumulation of material happens: the flow in and out is equal. In that case, $X_j = Y_j$. But more generally, there is a *stock accumulation* in the process of $\Delta_j = X_j - Y_j$.

## Mass balance of objects

Conservation of mass should apply for the intermediate object types for which a balancing "market" process is needed:

$$
\sum_j s_{ij} = \sum_j u_{ij}
$$

Conversely, other objects can be treated as "external": they can be consumed and produced by processes freely, and their consumption and production do not need to be balanced within the model system boundary.

## "Pulling" and "pushing" flows

To define the equations making up the model, usually there is some part of the system where we wish to start by specifying the flows using known data, or user-controlled parameters. Then, the model should propagate these specified flows through the system, by:

- Where multiple processes can produce the same object, specifying the relative fractions of demand which should be allocated to each process. For a total demand $D_i$, supply from each possible process $j$ will be:

  $$
  s_{ij} = \alpha_{ij} D_i
  $$
  where $\sum_j \alpha_{ij} = 1$.

- Where the supply from a process $s_{ij}$, has been determined, propagate this to the total output magnitude of the process:

  $$
  Y_j = \frac{s_{ij}}{S_{ij}}
  $$
  
- Where the process has a known (possibly zero) stock accumulation $\Delta_j$, find the total input magnitude of the process:

  $$ X_j = Y_j + \Delta_j $$

  and hence the required use of objects into the process

  $$
  u_{ij} = U_{ij} X_j
  $$
  
- Where the objects should have a balancing market in the model, these new use requirements lead to new demand for supply of the objects:

  $$
  D_i = \sum_j u_{ij}
  $$
  
- These newly-determined further demands for new objects, which can further propagate by repeating these steps (while taking care to deal with the possibility of loops).

Left unrestricted, this propagation will pass through the model until objects are reached which are external inputs (no processes are defined which produce them). Sometimes this is not desired. For example, in a typical material flow model, a stock model will determine demand for new material as well as availability of end-of-life material to be recycled. At some point in the model, there will be a balancing point where supply of recycled material is balanced against demand for new material, with the shortfall being made up from primary production. So, when the demand for new material is "pulled" through the downstream stages of the model, it should reach only as far upstream as this balancing point.

The above steps are written from the perspective of "pulling" demand through the model to cause upstream production, but the opposite applies similarly where supply is "pushed" through the model to cause downstream consumption.

## Implementation using `flowprog`

`flowprog` essentially provides three things:

- Book-keeping for the current values of system variables ($X_j, Y_j$) and the history of modelling steps that contributed to them.
- Functions to compute a full set of flows, caused by propagating supply/demand for an object in one part of the model a series of processes and "markets".
- Functions to query the current values, e.g. to find out how much additional production of an object is required to balance its market.

By computing partial sets of flows, perhaps based on user-specified values or data, and on querying the current interim state of the model, a full model can be built up in simple steps. The following sections describe in greater deal the specific functions that allow you to do this.

Because `flowprog` works with symbolic expressions (equations), the model can be *defined* and tested as a separate step, and then *evaluated* as many times as needed, e.g. to test different parameter values, fit to different countries' data, or to evaluate uncertainty using Monte Carlo sampling.

## Incremental Model Building with `add()`

The core workflow of flowprog is **incremental model building**. Unlike traditional models where all equations are specified simultaneously, flowprog models are built step-by-step by adding flows to the model state.

### The `add()` method

The `add()` method accumulates calculated flows into the model:

```python
model.add(flows_dict, label="Description of this step")
```

- `flows_dict`: Dictionary mapping symbols (like `X[j]` or `Y[j]`) to their values/expressions
- `label`: Optional string describing what this step represents (used for history tracking)

### Example workflow

```python
# Step 1: Add demand-driven flows
steel_flows = model.pull_production("Steel", demand_value, until_objects=["Iron ore"])
model.add(steel_flows, label="Steel production from demand")

# Step 2: Add supply from renewable sources
renewable_flows = model.pull_process_output("SolarPanel", "Electricity", capacity)
model.add(renewable_flows, label="Solar electricity supply")

# Step 3: Fill remaining demand with backup sources
gap = model.object_production_deficit("Electricity")
backup_flows = model.pull_production("Electricity", gap, ...)
model.add(backup_flows, label="Fossil fuel backup generation")
```

### History tracking

Each time you call `add()`, the label is recorded in the model's history. You can inspect how any variable was calculated:

```python
model.get_history(model.Y[j])  # Returns list of labels that contributed to Y[j]
```

This is invaluable for debugging and understanding complex models. The energy example demonstrates this in the notebook output.

## Multi-Step Logic and Balancing

One of flowprog's key capabilities is **multi-step logic**: building models in phases where each phase checks the current state before deciding what to add next.

### Checking object balances

Three methods help you query the current supply/demand balance:

1. **`object_balance(object_id)`**: Returns `(production - consumption)` for an object
   - Positive: excess production
   - Negative: unmet demand

2. **`object_production_deficit(object_id)`**: Returns `Max(0, consumption - production)`
   - How much additional production is needed
   - Zero if production already meets or exceeds consumption

3. **`object_consumption_deficit(object_id)`**: Returns `Max(0, production - consumption)`
   - How much excess production exists
   - Zero if consumption matches or exceeds production

### Pattern: "Use preferred source first, then backup"

This pattern is common in energy systems and circular economy models:

```python
# Step 1: Try to meet demand with preferred source (e.g., recycled material, renewables)
preferred_flows = model.pull_production("Material", preferred_capacity, ...)
model.add(preferred_flows, label="Preferred source")

# Step 2: Check if there's still unmet demand
deficit = model.object_production_deficit("Material")

# Step 3: Fill the gap with backup source (e.g., virgin material, fossil fuels)
backup_flows = model.pull_production("Material", deficit, ...)
model.add(backup_flows, label="Backup source")
```

### Example from the energy model

The energy example demonstrates this pattern (docs/examples/energy/load_model.py:39-59):

```python
# First choice: Wind turbines (with capacity limit)
wt_supply = model.pull_process_output("WindTurbine", "Electricity", wind_capacity)
wt_supply_limited = model.limit(wt_supply, ...)
model.add(wt_supply_limited, label="Supply from wind turbines (first choice)")

# Second choice: Fill remaining demand with CCGT
model.add(
    model.pull_process_output("CCGT", "Electricity",
                             model.object_production_deficit("Electricity")),
    label="Supply from CCGT (second choice)"
)
```

This creates **discontinuous behavior**: as electricity demand increases, wind supplies everything until it hits capacity, then CCGT switches on to fill the gap.

## Capacity Limits and Merit-Order Dispatch

Capacity limits are essential for modeling real-world systems where production options are constrained. The `limit()` method enables **merit-order dispatch**: prioritizing options in sequence.

### The `limit()` method

```python
limited_flows = model.limit(proposed_flows, expr=target_expr, limit=limit_value)
```

- `proposed_flows`: Dictionary of flows you want to add (from pull/push methods)
- `expr`: Symbolic expression to constrain (e.g., total output of a process)
- `limit`: Maximum allowed value for `expr`

The method scales down the proposed flows as needed so that `expr` doesn't exceed `limit`.

### How it works

The `limit()` method uses a Piecewise function with three cases:

1. **Already at capacity**: `current >= limit` → add nothing (0)
2. **Proposed within limit**: `proposed <= limit` → add full proposed value
3. **Proposed exceeds limit**: Scale down by `(limit - current) / (proposed - current)`

This creates a smooth transition from "unconstrained" to "fully constrained" behavior.

### Example: Wind turbine capacity

From the energy example:

```python
# Propose wind supply up to installed capacity
wt_supply = model.pull_process_output("WindTurbine", "Electricity",
                                      sy.Symbol("S_1"))  # Installed capacity

# But don't generate more than current electricity demand
wt_supply_limited = model.limit(
    wt_supply,
    expr=model.expr("ProcessOutput", process_id="WindTurbine", object_id="Electricity"),
    limit=model.expr("Consumption", object_id="Electricity")
)

model.add(wt_supply_limited, label="Wind supply (limited by demand)")
```

This ensures wind turbines never generate more electricity than is currently needed, even if their installed capacity is higher.

### Pattern: Cascading priorities

You can chain multiple capacity-limited sources to create a merit order:

```python
# Priority 1: Cheapest/cleanest option (capacity limited)
option1_flows = model.limit(
    model.pull_production("Product", capacity1, ...),
    expr=..., limit=capacity1
)
model.add(option1_flows, label="Priority 1")

# Priority 2: More expensive option (fills remaining gap)
deficit1 = model.object_production_deficit("Product")
option2_flows = model.limit(
    model.pull_production("Product", capacity2, ...),
    expr=..., limit=capacity2
)
model.add(option2_flows, label="Priority 2")

# Priority 3: Most expensive backup (no limit)
deficit2 = model.object_production_deficit("Product")
option3_flows = model.pull_production("Product", deficit2, ...)
model.add(option3_flows, label="Priority 3 (backup)")
```

This creates realistic dispatch behavior where cheaper/cleaner options are fully utilized before more expensive alternatives are deployed.

## Stock Accumulation

In many processes, material flows in and out at equal rates: $X_j = Y_j$. But some processes **accumulate stock**, where inputs can exceed outputs.

### When to use `has_stock=True`

Set `has_stock=True` for a process when:

- **Stock-holding processes**: Warehouses, storage facilities, landfills
- **Stock changes**: Growing/shrinking inventories (e.g., building up strategic reserves)
- **End-of-life stocks**: Products in use where additions ≠ removals

### Effect on model behavior

When `has_stock=False` (the default):
- Setting process output $Y_j$ automatically sets input $X_j = Y_j$
- Pull/push operations propagate through both input and output sides

When `has_stock=True`:
- Setting $Y_j$ does NOT automatically set $X_j$ (and vice versa)
- You must explicitly specify the stock change: $\Delta_j = X_j - Y_j$
- This allows inputs to exceed outputs (stock accumulation) or vice versa (stock depletion)

### Example: Stock-driven circular economy model

```python
# Process representing products in use
in_use_stock = Process(
    id="ProductsInUse",
    consumes=["NewProducts"],
    produces=["EndOfLifeProducts"],
    has_stock=True
)

# New product additions (input side)
model.add(
    model.push_process_input("ProductsInUse", "NewProducts", additions),
    label="Product additions to stock"
)

# End-of-life removals (output side) - may differ from additions
model.add(
    model.pull_process_output("ProductsInUse", "EndOfLifeProducts", removals),
    label="Products removed from use"
)

# The stock change is: additions - removals
```

This pattern is central to dynamic material flow models where stock changes drive system behavior, as described in the paper's petrochemical case study.
