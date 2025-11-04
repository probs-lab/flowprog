# MILP Transformation for Flowprog Models

## Overview

The MILP transformation module enables global optimization of flowprog models by converting them into Mixed Integer Linear/Quadratic Programs (MILP/MIQP). This is particularly valuable for **model calibration** where you have observed data and want to find the optimal parameters that best fit the observations.

## Why MILP for Calibration?

Traditional gradient-based optimization methods struggle with flowprog models because:

1. **Piecewise discontinuities** from `max()`, `min()`, and `Piecewise()` operations confuse gradient solvers
2. **Boundary behavior** is difficult to explore (solutions often lie at constraint boundaries)
3. **Local optima** can trap solvers, preventing global optimization
4. **No quality guarantees** - you don't know how far from optimal your solution is

MILP transformation solves these problems by:

- ✅ **Global optimality**: Proven optimal solution or certified bounds
- ✅ **Exact piecewise handling**: Linearizes max/min/abs operations correctly
- ✅ **Boundary exploration**: Naturally explores constraint boundaries
- ✅ **Quality bounds**: MIP gap tells you solution quality
- ✅ **Multiple solutions**: Can enumerate alternative near-optimal solutions

## Installation

Install flowprog with MILP support:

```bash
pip install flowprog[milp]
```

Or if using Poetry:

```bash
poetry install --extras milp
```

This installs the [Python-MIP](https://www.python-mip.com/) library, which provides access to:
- **CBC** (open-source, included)
- **Gurobi** (commercial, requires license)
- **CPLEX** (commercial, requires license)

## Quick Start

```python
from flowprog import Model, MILPTransformer
from flowprog.milp_solvers import PythonMIPBackend, SolverConfig
import sympy as sy

# 1. Create your flowprog model
model = Model(processes=[...], objects=[...])

# Build model with symbolic expressions
# ... use model.pull_production(), model.add(), etc.

# 2. Define optimization targets (observed data)
targets = {
    model.expr("SoldProduction", object_id="Steel"): 100.0,
    model.expr("Consumption", object_id="Electricity"): 65.0,
}

weights = {
    model.expr("SoldProduction", object_id="Steel"): 1.0,
    model.expr("Consumption", object_id="Electricity"): 1.0,
}

# 3. Transform to MILP
transformer = MILPTransformer(model)

milp_model = transformer.transform(
    objective_targets=targets,
    objective_weights=weights,
    variable_bounds={
        sy.Symbol("demand"): (0.0, 200.0),
        model.X[0]: (0.0, 100.0),
        # ... bounds for other variables
    }
)

# 4. Solve
backend = PythonMIPBackend()
config = SolverConfig(time_limit=60.0, mip_gap=0.01)
solution = backend.solve(milp_model, config)

# 5. Extract results
if solution.status == "optimal":
    print(f"Objective: {solution.objective_value}")
    original_vars = transformer.extract_solution(solution.variables)
    print(f"Optimal demand: {original_vars[sy.Symbol('demand')]}")
```

## How It Works

### 1. Expression Walking

The transformer walks through your flowprog model's symbolic expressions in topological order, processing:
- Decision variables (X, Y from processes)
- Intermediate expressions (from `model._intermediates`)
- Main expressions (from `model._values`)

### 2. Piecewise Linearization

Each piecewise operation is converted to MILP constraints:

#### max(0, expr)

For `z = max(0, a^T x + b)`, creates:
```
z >= 0
z >= a^T x + b
z <= (a^T x + b) + M*(1-y)
z <= M*y
y ∈ {0,1}
```

Where:
- `z`: auxiliary continuous variable for result
- `y`: binary indicator (1 if expr > 0, 0 otherwise)
- `M`: big-M constant computed from variable bounds

#### max(expr1, expr2)

For `z = max(e1, e2)`, creates:
```
z >= e1
z >= e2
z <= e1 + M1*(1-y)
z <= e2 + M2*y
y ∈ {0,1}
```

#### min(expr1, expr2)

Uses the identity: `min(a, b) = -max(-a, -b)`

#### abs(expr)

Uses the identity: `abs(x) = max(x, -x)`

#### General Piecewise (N branches)

For `Piecewise((val1, cond1), (val2, cond2), ..., (valN, True))`, creates:

```
# Create binary indicators (one per branch)
y1, y2, ..., yN ∈ {0,1}

# Exactly one branch must be active
y1 + y2 + ... + yN = 1

# For each branch i: if yi=1, then result = vali
result >= vali - M*(1 - yi)
result <= vali + M*(1 - yi)

# Optional: encode conditions as logical constraints
# e.g., if cond1 is "x >= limit":
#   x >= limit - M*(1 - y1)
```

Where:
- Binary `yi` indicates if branch `i` is active
- Result equals the value of whichever branch has `yi = 1`
- Optional condition constraints help guide the solver

**This handles the `limit()` function**, which creates 3-branch Piecewise expressions:
```python
Piecewise(
    (0, current >= limit),                          # Already at capacity
    (v, proposed <= limit),                         # Under capacity
    ((limit - current)/(proposed - current) * v, True)  # Scale down
)
```

### 3. Bounds Analysis

Critical for computing tight big-M values:

```python
from flowprog.milp_transform import BoundsAnalyzer

analyzer = BoundsAnalyzer({
    x: (0.0, 10.0),
    y: (-5.0, 5.0),
})

# Compute bounds on complex expression
lower, upper = analyzer.get_bounds(2*x + 3*y - 10)
# Returns: (-25.0, 25.0)

# Get appropriate big-M
M = analyzer.compute_big_m(2*x + 3*y - 10)
# Returns: 27.5 (with 10% slack)
```

### 4. Objective Function

For calibration, the objective minimizes weighted sum of squared errors:

```
minimize: Σ w_i * (output_i - target_i)²
```

This expands to a **Mixed Integer Quadratic Program (MIQP)**:

```
minimize: Σ w_i * output_i² - 2*w_i*target_i*output_i + w_i*target_i²
```

The constant term `w_i*target_i²` is omitted as it doesn't affect optimization.

## Advanced Usage

### Multiple Solutions

Find alternative near-optimal solutions:

```python
backend = PythonMIPBackend()

solutions = backend.find_multiple_solutions(
    milp_model,
    num_solutions=5,
    tolerance=0.05,  # Within 5% of optimal
    config=SolverConfig(verbose=False)
)

for i, sol in enumerate(solutions):
    print(f"Solution {i+1}: objective = {sol.objective_value}")
    vars = transformer.extract_solution(sol.variables)
    # Analyze alternative solution
```

### Export to Dictionary

For integration with other solvers:

```python
from flowprog.milp_solvers import DictExportBackend

backend = DictExportBackend()
model_dict = backend.export_to_dict(milp_model)

# model_dict contains:
# - "variables": list of variable definitions
# - "constraints": list of constraint definitions
# - "objective": objective function definition
# - "statistics": problem size stats
```

### Custom Bounds and Constraints

```python
# Define variable bounds
variable_bounds = {
    model.X[j]: (0.0, capacity[j]) for j in range(num_processes)
}

# Add custom fixed values (recipe coefficients)
fixed_values = {
    model.S[i, j]: recipe_data[(i, j)]
    for i in range(num_objects)
    for j in range(num_processes)
}

milp_model = transformer.transform(
    objective_targets=targets,
    variable_bounds=variable_bounds,
    fixed_values=fixed_values
)

# Add custom constraints after transformation
from flowprog.milp_transform import MILPConstraint, ConstraintType

milp_model.add_constraint(MILPConstraint(
    name="custom_constraint",
    coefficients={"var1": 1.0, "var2": 1.0},
    constraint_type=ConstraintType.LINEAR_EQ,
    rhs=1.0,
    description="var1 + var2 = 1"
))
```

## Solver Configuration

```python
config = SolverConfig(
    time_limit=300.0,      # Max 5 minutes
    mip_gap=0.01,          # 1% optimality gap
    threads=4,             # Use 4 CPU cores
    verbose=True,          # Show solver output
    solver_name="cbc"      # Or "gurobi", "cplex"
)

solution = backend.solve(milp_model, config)
```

## Problem Size Considerations

Typical calibration problems:

| Component | Small | Medium | Large |
|-----------|-------|--------|-------|
| Continuous variables | 10-20 | 50-100 | 200+ |
| Binary variables | 5-10 | 20-50 | 100+ |
| Constraints | 20-50 | 100-300 | 1000+ |
| Solve time (CBC) | <1s | 1-60s | 1-10min |
| Solve time (Gurobi) | <1s | <10s | 10s-2min |

**Tips for scaling:**

1. **Tight bounds**: Provide accurate variable bounds for smaller big-M values
2. **Warm start**: Use good initial solutions when available
3. **MIP gap**: Accept 1-5% gap for faster solutions
4. **Commercial solvers**: Gurobi/CPLEX are 10-100x faster than CBC for large problems

## Troubleshooting

### Infeasible Problem

```
Status: infeasible
```

**Causes:**
- Incompatible constraints (targets cannot be met)
- Too-tight variable bounds
- Conflicting fixed values

**Solutions:**
1. Relax variable bounds
2. Check constraint compatibility
3. Use penalty-based objective instead of hard constraints

### Slow Solve Times

**Solutions:**
1. Tighten variable bounds (enables smaller big-M)
2. Reduce MIP gap tolerance
3. Upgrade to Gurobi/CPLEX
4. Simplify model (fewer piecewise operations)

### Numerical Issues

```
Warning: Large big-M values detected
```

**Causes:**
- Unbounded variables
- Wide variable ranges

**Solutions:**
1. Provide explicit bounds for all variables
2. Normalize/scale variables to similar ranges
3. Use tighter bounds based on domain knowledge

### Unsupported Expressions

**Current limitations:**
- Linear expressions only (no x*y products between variables)
- Piecewise conditions must be linear inequalities or True
- Complex logical combinations (AND/OR) in Piecewise conditions may be ignored

**Supported:**
- ✅ General N-branch Piecewise expressions (including `limit()` function)
- ✅ Max, Min, Abs operations
- ✅ Linear inequalities as Piecewise conditions (x >= y, x <= y)
- ✅ Nested piecewise (each branch can contain sub-expressions)

**Workarounds for limitations:**
1. For bilinear terms (x*y): Use McCormick envelopes (not yet implemented)
2. For complex logic: Reformulate as multiple simpler piecewise expressions
3. Use symbolic substitution to simplify before transformation

## Architecture

```
flowprog.Model
    ↓
MILPTransformer
    ├── BoundsAnalyzer (compute big-M values)
    ├── PiecewiseLinearizer (create MILP constraints)
    └── MILPModel (solver-agnostic representation)
        ↓
SolverBackend
    ├── PythonMIPBackend (CBC/Gurobi/CPLEX)
    └── DictExportBackend (export to dict)
        ↓
SolverSolution
    ↓
extract_solution() → Original variables
```

## API Reference

### MILPTransformer

```python
class MILPTransformer:
    def __init__(self, flowprog_model: Model)

    def transform(
        self,
        objective_targets: Dict[sy.Expr, float],
        objective_weights: Optional[Dict[sy.Expr, float]] = None,
        variable_bounds: Optional[Dict[sy.Symbol, Tuple[float, float]]] = None,
        fixed_values: Optional[Dict[sy.Symbol, float]] = None
    ) -> MILPModel

    def extract_solution(
        self,
        solution: Dict[str, float]
    ) -> Dict[sy.Symbol, float]
```

### BoundsAnalyzer

```python
class BoundsAnalyzer:
    def __init__(self, variable_bounds: Dict[sy.Symbol, Tuple[float, float]])

    def get_bounds(self, expr: sy.Expr) -> Tuple[float, float]

    def compute_big_m(self, expr: sy.Expr, slack: float = 1.1) -> float
```

### SolverBackend

```python
class SolverBackend(ABC):
    @abstractmethod
    def solve(
        self,
        milp_model: MILPModel,
        config: Optional[SolverConfig] = None
    ) -> SolverSolution

class PythonMIPBackend(SolverBackend):
    def find_multiple_solutions(
        self,
        milp_model: MILPModel,
        num_solutions: int,
        tolerance: Optional[float] = None,
        config: Optional[SolverConfig] = None
    ) -> List[SolverSolution]
```

### SolverConfig

```python
@dataclass
class SolverConfig:
    time_limit: Optional[float] = None
    mip_gap: Optional[float] = None
    threads: Optional[int] = None
    verbose: bool = True
    solver_name: Optional[str] = None
```

### SolverSolution

```python
@dataclass
class SolverSolution:
    status: str  # "optimal", "feasible", "infeasible", "unbounded"
    objective_value: Optional[float] = None
    variables: Dict[str, float] = None
    gap: Optional[float] = None
    solve_time: Optional[float] = None
```

## Examples

See `examples/milp_calibration_example.py` for a complete working example with:
- Model construction
- Target definition
- MILP transformation
- Solving
- Solution interpretation
- Multiple solution enumeration

## Limitations and Future Work

**Current limitations:**
- Linear expressions only (no bilinear terms x*y between decision variables)
- Piecewise conditions must be linear inequalities (complex AND/OR logic not fully supported)
- Big-M formulation (SOS constraints could be more efficient)

**Planned enhancements:**
- SOS (Special Ordered Sets) constraints for tighter LP relaxations
- Indicator constraints (native piecewise support in Gurobi/CPLEX)
- Automatic symmetry breaking
- McCormick envelope for bilinear terms
- Warm starting from heuristic solutions
- Sensitivity analysis tools

## References

- [Python-MIP Documentation](https://docs.python-mip.com/)
- [MILP Modeling Best Practices](https://www.gurobi.com/resources/mixed-integer-programming-mip-a-primer-on-the-basics/)
- [Big-M Method](https://optimization.cbe.cornell.edu/index.php?title=Disjunctive_inequalities)
- [Piecewise Linear Approximation](https://or.stackexchange.com/questions/tagged/piecewise-linear)

## License

MIT License - Same as flowprog
