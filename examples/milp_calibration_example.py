"""
Example: MILP-based Model Calibration

This example demonstrates how to use MILP transformation to calibrate a flowprog model
against observed data. The MILP formulation provides:

1. Global optimality guarantees
2. Robust boundary handling
3. Multiple solution enumeration
4. Quality bounds (optimality gap)

Scenario:
We have a simple production system with multiple processes that can produce the same
outputs. We observe actual production levels and want to find the optimal allocation
of process activities that best matches the observations while respecting physical
constraints (max operations, capacity limits, etc.).
"""

import sympy as sy
from flowprog.imperative_model import Model, Process, Object
from flowprog.milp_transform import MILPTransformer
from flowprog.milp_solvers import get_recommended_backend, PythonMIPBackend, SolverConfig
from rdflib import URIRef


def create_example_model():
    """
    Create a simple flowprog model for demonstration.

    The model has:
    - Two processes that can produce Steel (EAF and H2DRI)
    - Electricity is consumed by both processes
    - Additional electricity from renewable sources (limited capacity)
    - Deficit filled by gas-based generation
    """
    # Create processes
    processes = [
        Process(id="SteelProductionEAF", produces=["Steel"], consumes=["Electricity"]),
        Process(id="SteelProductionH2DRI", produces=["Steel"], consumes=["Electricity"]),
        Process(id="WindTurbine", produces=["Electricity"], consumes=[]),
        Process(id="CCGT", produces=["Electricity"], consumes=[]),
    ]

    # Create objects
    metric_uri = URIRef("http://qudt.org/vocab/quantitykind/Mass")
    objects = [
        Object(id="Steel", metric=metric_uri, has_market=False),
        Object(id="Electricity", metric=metric_uri, has_market=True),
    ]

    # Create model
    model = Model(processes=processes, objects=objects)

    # Define demand for steel
    steel_demand = sy.Symbol("steel_demand", nonnegative=True)

    # Pull steel production with allocation between processes
    # We'll use symbolic allocation that will be calibrated
    eaf_fraction = sy.Symbol("eaf_fraction", nonnegative=True)  # Decision variable
    h2dri_fraction = sy.Symbol("h2dri_fraction", nonnegative=True)  # Decision variable

    model.add(
        model.pull_production(
            "Steel",
            steel_demand,
            until_objects=["Electricity"],
            allocate_backwards={
                "Steel": {
                    "SteelProductionEAF": eaf_fraction,
                    "SteelProductionH2DRI": h2dri_fraction
                }
            }
        )
    )

    # Add renewable electricity with capacity limit
    renewable_capacity = sy.Symbol("renewable_capacity", nonnegative=True)

    renewable_supply = model.pull_process_output(
        "WindTurbine",
        "Electricity",
        renewable_capacity,
        until_objects=[]
    )

    # Limit renewable to consumption (using Piecewise)
    renewable_limited = model.limit(
        renewable_supply,
        expr=model.expr("ProcessOutput", process_id="WindTurbine", object_id="Electricity"),
        limit=model.expr("Consumption", object_id="Electricity")
    )

    model.add(renewable_limited)

    # Fill electricity deficit with gas generation
    # This uses Max(0, -balance) which is perfect for MILP
    model.add(
        model.pull_process_output(
            "CCGT",
            "Electricity",
            model.object_production_deficit("Electricity")
        )
    )

    return model


def define_recipe_data():
    """
    Define production coefficients (recipe data).

    Returns:
        Dict mapping indexed symbols to values
    """
    # S[i,j] = output of object i from process j per unit Y[j]
    # U[i,j] = input of object i to process j per unit X[j]

    return {
        # Steel production from EAF (index 0)
        "S[0,0]": 1.0,  # Steel output per unit
        "U[1,0]": 0.5,  # Electricity input per unit (efficient)

        # Steel production from H2DRI (index 1)
        "S[0,1]": 1.0,  # Steel output per unit
        "U[1,1]": 0.8,  # Electricity input per unit (less efficient)

        # Wind turbine (index 2)
        "S[1,2]": 1.0,  # Electricity output

        # CCGT gas plant (index 3)
        "S[1,3]": 1.0,  # Electricity output
    }


def run_calibration_example():
    """
    Main calibration workflow demonstrating MILP transformation.
    """
    print("=" * 80)
    print("MILP-Based Flowprog Model Calibration Example")
    print("=" * 80)
    print()

    # Step 1: Create model
    print("Step 1: Creating flowprog model...")
    model = create_example_model()
    print(f"  Model created with {len(model.processes)} processes and {len(model.objects)} objects")
    print()

    # Step 2: Define observed data (what we want to calibrate to)
    print("Step 2: Defining observed data for calibration...")

    # Suppose we observed:
    # - Total steel production: 100 units
    # - Total electricity consumption: 65 units
    # - Renewable electricity should be maximized (within capacity of 40 units)

    observed_data = {
        "steel_production": 100.0,
        "electricity_consumption": 65.0,
        "renewable_capacity_available": 40.0,
    }

    print(f"  Observed steel production: {observed_data['steel_production']}")
    print(f"  Observed electricity consumption: {observed_data['electricity_consumption']}")
    print(f"  Renewable capacity available: {observed_data['renewable_capacity_available']}")
    print()

    # Step 3: Set up MILP transformation
    print("Step 3: Setting up MILP transformation...")

    # Define what we want to optimize (minimize squared errors)
    steel_prod_expr = model.expr("SoldProduction", object_id="Steel")
    elec_cons_expr = model.expr("Consumption", object_id="Electricity")

    objective_targets = {
        steel_prod_expr: observed_data["steel_production"],
        elec_cons_expr: observed_data["electricity_consumption"],
    }

    # Equal weights for both objectives
    objective_weights = {
        steel_prod_expr: 1.0,
        elec_cons_expr: 1.0,
    }

    # Define variable bounds
    steel_demand = sy.Symbol("steel_demand", nonnegative=True)
    eaf_fraction = sy.Symbol("eaf_fraction", nonnegative=True)
    h2dri_fraction = sy.Symbol("h2dri_fraction", nonnegative=True)
    renewable_capacity = sy.Symbol("renewable_capacity", nonnegative=True)

    # Get X and Y symbols for each process
    X = model.X
    Y = model.Y

    variable_bounds = {
        steel_demand: (0.0, 200.0),
        eaf_fraction: (0.0, 1.0),
        h2dri_fraction: (0.0, 1.0),
        renewable_capacity: (0.0, observed_data["renewable_capacity_available"]),

        # Process activity bounds (generous to allow exploration)
        X[0]: (0.0, 150.0),  # EAF input
        Y[0]: (0.0, 150.0),  # EAF output
        X[1]: (0.0, 150.0),  # H2DRI input
        Y[1]: (0.0, 150.0),  # H2DRI output
        X[2]: (0.0, 50.0),   # Wind input
        Y[2]: (0.0, 50.0),   # Wind output
        X[3]: (0.0, 100.0),  # CCGT input
        Y[3]: (0.0, 100.0),  # CCGT output
    }

    # Additional constraint: allocation fractions must sum to 1
    # We'll add this as a fixed value for simplicity
    # In practice, this would be a constraint: eaf_fraction + h2dri_fraction = 1

    print("  Objective: Minimize squared errors in steel production and electricity consumption")
    print(f"  Decision variables: {len(variable_bounds)}")
    print()

    # Step 4: Transform to MILP
    print("Step 4: Transforming to MILP formulation...")

    transformer = MILPTransformer(model)

    # Note: We need to handle the allocation constraint
    # For this example, let's simplify by fixing it
    # In a real scenario, you'd add it as an explicit constraint

    milp_model = transformer.transform(
        objective_targets=objective_targets,
        objective_weights=objective_weights,
        variable_bounds=variable_bounds,
        fixed_values={}  # Recipe coefficients would go here
    )

    # Add allocation constraint: eaf_fraction + h2dri_fraction = 1
    # This is a linear constraint
    from flowprog.milp_transform import MILPConstraint, ConstraintType

    eaf_var_name = milp_model.variable_mapping.get(eaf_fraction, "eaf_fraction")
    h2dri_var_name = milp_model.variable_mapping.get(h2dri_fraction, "h2dri_fraction")

    if eaf_var_name in milp_model.variables and h2dri_var_name in milp_model.variables:
        milp_model.add_constraint(MILPConstraint(
            name="allocation_sum",
            coefficients={eaf_var_name: 1.0, h2dri_var_name: 1.0},
            constraint_type=ConstraintType.LINEAR_EQ,
            rhs=1.0,
            description="EAF and H2DRI fractions must sum to 1"
        ))

    stats = milp_model.get_statistics()
    print("  MILP Model Statistics:")
    for key, value in stats.items():
        print(f"    {key}: {value}")
    print()

    # Step 5: Solve with MILP solver
    print("Step 5: Solving MILP problem...")

    # Get solver backend
    try:
        backend = PythonMIPBackend()
        print("  Using Python-MIP backend with CBC solver")
    except ImportError:
        backend = get_recommended_backend()
        print(f"  Using {backend.get_solver_name()} backend")

    config = SolverConfig(
        time_limit=60.0,  # 60 seconds max
        mip_gap=0.01,     # 1% optimality gap tolerance
        verbose=True
    )

    solution = backend.solve(milp_model, config)

    print()
    print("Step 6: Solution Results")
    print("-" * 80)

    if solution.status in ["optimal", "feasible"]:
        print(f"  Status: {solution.status.upper()}")
        print(f"  Objective value: {solution.objective_value:.4f}")

        if solution.gap is not None:
            print(f"  MIP gap: {solution.gap * 100:.2f}%")

        if solution.solve_time is not None:
            print(f"  Solve time: {solution.solve_time:.2f} seconds")

        print()

        # Extract solution for original variables
        original_solution = transformer.extract_solution(solution.variables)

        print("  Optimal Decision Variables:")
        print(f"    Steel demand: {original_solution.get(steel_demand, 'N/A'):.2f}")
        print(f"    EAF fraction: {original_solution.get(eaf_fraction, 'N/A'):.4f}")
        print(f"    H2DRI fraction: {original_solution.get(h2dri_fraction, 'N/A'):.4f}")
        print(f"    Renewable capacity used: {original_solution.get(renewable_capacity, 'N/A'):.2f}")
        print()

        print("  Process Activities:")
        for j in range(4):
            x_val = original_solution.get(X[j], 0.0)
            y_val = original_solution.get(Y[j], 0.0)
            process_name = model.processes[j].id
            print(f"    {process_name}: X={x_val:.2f}, Y={y_val:.2f}")

        print()

        # Calculate achieved outputs
        recipe_data = define_recipe_data()

        # Convert recipe data string keys to actual values
        # This is simplified; in practice you'd use model.eval() with proper substitution
        print("  Model Outputs vs. Observations:")

        # For demonstration, manually calculate based on solution
        # In practice, you'd evaluate the expressions with the solution

        total_steel = sum(
            original_solution.get(Y[j], 0.0) * recipe_data.get(f"S[0,{j}]", 0.0)
            for j in range(2)  # Only steel-producing processes
        )

        total_elec_consumed = sum(
            original_solution.get(X[j], 0.0) * recipe_data.get(f"U[1,{j}]", 0.0)
            for j in range(2)  # Only steel processes consume electricity
        )

        total_elec_produced = sum(
            original_solution.get(Y[j], 0.0) * recipe_data.get(f"S[1,{j}]", 0.0)
            for j in [2, 3]  # Wind and CCGT
        )

        print(f"    Steel production: {total_steel:.2f} (target: {observed_data['steel_production']})")
        print(f"    Electricity consumption: {total_elec_consumed:.2f} (target: {observed_data['electricity_consumption']})")
        print(f"    Electricity production: {total_elec_produced:.2f}")

        print()
        print("  Insights:")
        eaf_frac = original_solution.get(eaf_fraction, 0.0)
        h2dri_frac = original_solution.get(h2dri_fraction, 0.0)

        if eaf_frac > h2dri_frac:
            print(f"    - EAF is preferred ({eaf_frac*100:.1f}%) due to lower electricity intensity")
        else:
            print(f"    - H2DRI is preferred ({h2dri_frac*100:.1f}%) despite higher electricity use")

        renewable_used = original_solution.get(renewable_capacity, 0.0)
        renewable_available = observed_data["renewable_capacity_available"]

        if renewable_used >= renewable_available * 0.95:
            print(f"    - Renewable capacity fully utilized ({renewable_used:.1f}/{renewable_available:.1f})")
        else:
            print(f"    - Renewable capacity partially used ({renewable_used:.1f}/{renewable_available:.1f})")

    else:
        print(f"  Status: {solution.status.upper()}")
        print("  No solution found. This could indicate:")
        print("    - Infeasible constraints (targets cannot be met)")
        print("    - Unbounded problem (need tighter bounds)")
        print("    - Numerical issues (check big-M values)")

    print()
    print("=" * 80)
    print("Example completed.")
    print()
    print("Key Advantages of MILP Approach:")
    print("  1. Global optimality guarantee (within MIP gap)")
    print("  2. Handles piecewise operations (max, min) exactly")
    print("  3. Can explore boundary cases that confuse gradient-based solvers")
    print("  4. Provides quality bounds (gap) even for non-optimal solutions")
    print("  5. Can enumerate multiple near-optimal solutions")
    print("=" * 80)


if __name__ == "__main__":
    # Run the example
    try:
        run_calibration_example()
    except Exception as e:
        print(f"\nError running example: {e}")
        print("\nThis might be because:")
        print("  - Python-MIP is not installed (pip install mip)")
        print("  - Model construction has issues")
        print("  - MILP transformation encountered unsupported expressions")
        import traceback
        traceback.print_exc()
