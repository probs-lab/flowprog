"""
Integration tests for MILP transformation and solving.

These tests verify that the complete pipeline (model -> MILP -> solve -> extract)
produces correct results by:
1. Creating a flowprog model
2. Transforming it to MILP
3. Solving with a backend
4. Validating the solution matches expected values
"""

import pytest
import sympy as sy
from flowprog.imperative_model import Model, Process, Object
from flowprog.milp_transform import MILPTransformer
from rdflib import URIRef

# Check if mip is available for solving
try:
    from flowprog.milp_solvers import PythonMIPBackend, SolverConfig
    HAS_MIP = True
except ImportError:
    HAS_MIP = False

pytestmark = pytest.mark.skipif(not HAS_MIP, reason="Python-MIP not installed")


class TestMILPIntegration:
    """Integration tests that solve actual MILP problems and verify results."""

    def test_consistent_targets_exact_match(self):
        """
        Test with internally consistent targets - solution should match exactly.

        Model: Single process producing Steel from Electricity
        Target: 100 units of Steel
        Expected: Process should produce exactly 100 units
        """
        # Create simple model
        processes = [
            Process(id="SteelProduction", produces=["Steel"], consumes=["Electricity"])
        ]
        objects = [
            Object(id="Steel", metric=URIRef("http://qudt.org/vocab/quantitykind/Mass")),
            Object(id="Electricity", metric=URIRef("http://qudt.org/vocab/quantitykind/Energy"))
        ]

        model = Model(processes=processes, objects=objects)

        # Define demand
        steel_demand = sy.Symbol("steel_demand", nonnegative=True)

        # Pull production
        model.add(model.pull_production("Steel", steel_demand, until_objects=["Electricity"]))

        # Add electricity supply (unconstrained for this test)
        elec_supply = sy.Symbol("elec_supply", nonnegative=True)
        model.add(model.push_consumption("Electricity", elec_supply, until_objects=[]))

        # Transform to MILP
        transformer = MILPTransformer(model)

        # Target: 100 units of steel production
        steel_production_expr = model.expr("SoldProduction", object_id="Steel")

        objective_targets = {
            steel_production_expr: 100.0
        }

        variable_bounds = {
            steel_demand: (0.0, 200.0),
            elec_supply: (0.0, 500.0),
            model.X[0]: (0.0, 200.0),
            model.Y[0]: (0.0, 200.0),
        }

        # Recipe: 1 unit Steel output per Y[0], 0.5 unit Electricity input per X[0]
        fixed_values = {
            model.S[0, 0]: 1.0,   # Steel output coefficient
            model.U[1, 0]: 0.5,   # Electricity input coefficient
        }

        milp_model = transformer.transform(
            objective_targets=objective_targets,
            variable_bounds=variable_bounds,
            fixed_values=fixed_values
        )

        # Solve
        backend = PythonMIPBackend()
        config = SolverConfig(verbose=False, time_limit=10.0)
        solution = backend.solve(milp_model, config)

        # Verify solution
        assert solution.status in ["optimal", "feasible"], f"Solution status: {solution.status}"

        # Extract original variables
        original_solution = transformer.extract_solution(solution.variables)

        # Verify steel production matches target (within tolerance)
        steel_demand_val = original_solution.get(steel_demand, 0.0)
        Y_0 = original_solution.get(model.Y[0], 0.0)

        # Steel production = Y[0] * S[0,0] = Y[0] * 1.0
        steel_production = Y_0 * 1.0

        assert abs(steel_production - 100.0) < 0.1, \
            f"Expected steel production ~100, got {steel_production}"

        # For consistent targets, objective should be near zero
        assert solution.objective_value < 0.1, \
            f"Expected objective near 0 for consistent targets, got {solution.objective_value}"

        # Verify electricity consumption matches production needs
        X_0 = original_solution.get(model.X[0], 0.0)
        elec_consumed = X_0 * 0.5

        # For balanced model, should have enough electricity
        elec_supply_val = original_solution.get(elec_supply, 0.0)
        assert elec_supply_val >= elec_consumed - 0.1, \
            f"Electricity supply ({elec_supply_val}) should cover consumption ({elec_consumed})"

    def test_inconsistent_targets_compromise(self):
        """
        Test with inconsistent targets - solution should find weighted compromise.

        Model: Process with fixed input/output ratio
        Targets: Conflicting values for input and output
        Expected: Solution between the two targets, weighted by objective weights
        """
        # Create simple model
        processes = [
            Process(id="Process1", produces=["Output"], consumes=["Input"])
        ]
        objects = [
            Object(id="Output", metric=URIRef("http://qudt.org/vocab/quantitykind/Mass")),
            Object(id="Input", metric=URIRef("http://qudt.org/vocab/quantitykind/Mass"))
        ]

        model = Model(processes=processes, objects=objects)

        # Add production and consumption
        output_demand = sy.Symbol("output_demand", nonnegative=True)
        model.add(model.pull_production("Output", output_demand, until_objects=["Input"]))

        input_supply = sy.Symbol("input_supply", nonnegative=True)
        model.add(model.push_consumption("Input", input_supply, until_objects=[]))

        # Transform to MILP
        transformer = MILPTransformer(model)

        # Conflicting targets:
        # - Want 100 units of output
        # - Want 30 units of input consumption
        # - But recipe requires 2:1 ratio (2 input per 1 output)
        # With equal weights, should settle around:
        #   Output: ~75, Input: ~60 (compromise between 100 and 30, respecting 2:1 ratio)

        output_expr = model.expr("SoldProduction", object_id="Output")
        input_expr = model.expr("Consumption", object_id="Input")

        objective_targets = {
            output_expr: 100.0,
            input_expr: 30.0,  # Inconsistent! Would need 200 input for 100 output
        }

        # Equal weights
        objective_weights = {
            output_expr: 1.0,
            input_expr: 1.0,
        }

        variable_bounds = {
            output_demand: (0.0, 200.0),
            input_supply: (0.0, 400.0),
            model.X[0]: (0.0, 200.0),
            model.Y[0]: (0.0, 200.0),
        }

        # Recipe: 1 output per Y[0], 2 input per X[0]
        fixed_values = {
            model.S[0, 0]: 1.0,   # Output coefficient
            model.U[1, 0]: 2.0,   # Input coefficient (2:1 ratio)
        }

        milp_model = transformer.transform(
            objective_targets=objective_targets,
            objective_weights=objective_weights,
            variable_bounds=variable_bounds,
            fixed_values=fixed_values
        )

        # Solve
        backend = PythonMIPBackend()
        config = SolverConfig(verbose=False, time_limit=10.0)
        solution = backend.solve(milp_model, config)

        assert solution.status in ["optimal", "feasible"]

        original_solution = transformer.extract_solution(solution.variables)

        Y_0 = original_solution.get(model.Y[0], 0.0)
        X_0 = original_solution.get(model.X[0], 0.0)

        output_production = Y_0 * 1.0
        input_consumption = X_0 * 2.0

        # Solution should be between the conflicting targets
        assert 20.0 < output_production < 110.0, \
            f"Output should be between targets, got {output_production}"
        assert 20.0 < input_consumption < 210.0, \
            f"Input should be between targets, got {input_consumption}"

        # Should respect the 2:1 ratio constraint
        # X[0] should equal Y[0] for balanced process
        assert abs(X_0 - Y_0) < 0.1, \
            f"Process should be balanced (X≈Y), got X={X_0}, Y={Y_0}"

        # Verify the ratio
        expected_input = output_production * 2.0
        assert abs(input_consumption - expected_input) < 0.1, \
            f"Input/output ratio should be 2:1, got input={input_consumption} for output={output_production}"

        # Objective should be positive (there's error)
        assert solution.objective_value > 1.0, \
            f"Expected positive objective for inconsistent targets, got {solution.objective_value}"

    def test_max_operation_in_calibration(self):
        """
        Test that Max operations are correctly handled in calibration.

        Model: Production with deficit filling (Max operation)
        Expected: Deficit should be correctly calculated and filled
        """
        # Create model with two processes
        processes = [
            Process(id="Primary", produces=["Product"], consumes=[]),
            Process(id="Backup", produces=["Product"], consumes=[]),
        ]
        objects = [
            Object(id="Product", metric=URIRef("http://qudt.org/vocab/quantitykind/Mass"))
        ]

        model = Model(processes=processes, objects=objects)

        # Primary production (limited)
        primary_amount = sy.Symbol("primary_amount", nonnegative=True)
        model.add(model.pull_process_output("Primary", "Product", primary_amount))

        # Backup fills deficit: Max(0, demand - primary)
        demand = sy.Symbol("demand", nonnegative=True)
        primary_production = model.expr("ProcessOutput", process_id="Primary", object_id="Product")

        # Create deficit: max(0, demand - primary_production)
        deficit = sy.Max(0, demand - primary_production, evaluate=False)
        model.add(model.pull_process_output("Backup", "Product", deficit))

        # Transform
        transformer = MILPTransformer(model)

        total_production_expr = model.expr("SoldProduction", object_id="Product")

        # Target: 100 total production
        objective_targets = {
            total_production_expr: 100.0
        }

        variable_bounds = {
            demand: (0.0, 200.0),
            primary_amount: (0.0, 40.0),  # Limited to 40
            model.X[0]: (0.0, 50.0),
            model.Y[0]: (0.0, 50.0),
            model.X[1]: (0.0, 150.0),
            model.Y[1]: (0.0, 150.0),
        }

        # Recipe: 1 output per process
        fixed_values = {
            model.S[0, 0]: 1.0,  # Primary output
            model.S[0, 1]: 1.0,  # Backup output
        }

        milp_model = transformer.transform(
            objective_targets=objective_targets,
            variable_bounds=variable_bounds,
            fixed_values=fixed_values
        )

        # Solve
        backend = PythonMIPBackend()
        config = SolverConfig(verbose=False, time_limit=10.0)
        solution = backend.solve(milp_model, config)

        assert solution.status in ["optimal", "feasible"]

        original_solution = transformer.extract_solution(solution.variables)

        Y_0 = original_solution.get(model.Y[0], 0.0)  # Primary production
        Y_1 = original_solution.get(model.Y[1], 0.0)  # Backup production

        total = Y_0 + Y_1

        # Should achieve target
        assert abs(total - 100.0) < 0.1, \
            f"Total production should be ~100, got {total}"

        # Primary should be at its limit (40)
        assert abs(Y_0 - 40.0) < 0.1, \
            f"Primary should be at limit (40), got {Y_0}"

        # Backup should fill the gap (60)
        assert abs(Y_1 - 60.0) < 0.1, \
            f"Backup should fill gap (60), got {Y_1}"

    def test_limit_function_with_capacity_constraint(self):
        """
        Test that limit() function correctly enforces capacity constraints.

        Model: Production limited by capacity using limit() function
        Expected: Production should not exceed capacity
        """
        # Create model
        processes = [
            Process(id="Producer", produces=["Product"], consumes=[])
        ]
        objects = [
            Object(id="Product", metric=URIRef("http://qudt.org/vocab/quantitykind/Mass"))
        ]

        model = Model(processes=processes, objects=objects)

        # Try to produce unlimited amount
        unlimited_amount = sy.Symbol("unlimited", nonnegative=True)
        production_values = model.pull_process_output("Producer", "Product", unlimited_amount)

        # Apply capacity limit using limit() function
        capacity = sy.Symbol("capacity", nonnegative=True)
        production_expr = model.expr("ProcessOutput", process_id="Producer", object_id="Product")

        limited_values = model.limit(
            values=production_values,
            expr=production_expr,
            limit=capacity
        )

        model.add(limited_values)

        # Transform
        transformer = MILPTransformer(model)

        production_total_expr = model.expr("SoldProduction", object_id="Product")

        # Target: Want 200 units, but capacity is only 100
        objective_targets = {
            production_total_expr: 200.0
        }

        variable_bounds = {
            unlimited: (0.0, 500.0),
            capacity: (100.0, 100.0),  # Fixed capacity
            model.X[0]: (0.0, 300.0),
            model.Y[0]: (0.0, 300.0),
        }

        fixed_values = {
            model.S[0, 0]: 1.0,  # Output coefficient
            capacity: 100.0,     # Capacity limit
        }

        milp_model = transformer.transform(
            objective_targets=objective_targets,
            variable_bounds=variable_bounds,
            fixed_values=fixed_values
        )

        # Solve
        backend = PythonMIPBackend()
        config = SolverConfig(verbose=False, time_limit=10.0)
        solution = backend.solve(milp_model, config)

        assert solution.status in ["optimal", "feasible"]

        original_solution = transformer.extract_solution(solution.variables)

        Y_0 = original_solution.get(model.Y[0], 0.0)
        production = Y_0 * 1.0

        # Production should be capped at capacity (100), not target (200)
        assert production <= 100.1, \
            f"Production should not exceed capacity (100), got {production}"

        # Should be at or near capacity
        assert production >= 99.0, \
            f"Production should be near capacity (100), got {production}"

        # Objective should be positive (couldn't reach target)
        assert solution.objective_value > 1.0, \
            f"Expected positive objective (couldn't reach target 200), got {solution.objective_value}"

    def test_weighted_objectives_balance(self):
        """
        Test that objective weights correctly balance conflicting goals.

        Different weights should produce different compromises.
        """
        def solve_with_weights(weight1, weight2):
            """Helper to solve with given weights."""
            processes = [
                Process(id="Process", produces=["Output"], consumes=["Input"])
            ]
            objects = [
                Object(id="Output", metric=URIRef("http://qudt.org/vocab/quantitykind/Mass")),
                Object(id="Input", metric=URIRef("http://qudt.org/vocab/quantitykind/Mass"))
            ]

            model = Model(processes=processes, objects=objects)

            output_demand = sy.Symbol("output_demand", nonnegative=True)
            model.add(model.pull_production("Output", output_demand, until_objects=["Input"]))

            input_supply = sy.Symbol("input_supply", nonnegative=True)
            model.add(model.push_consumption("Input", input_supply, until_objects=[]))

            transformer = MILPTransformer(model)

            output_expr = model.expr("SoldProduction", object_id="Output")
            input_expr = model.expr("Consumption", object_id="Input")

            objective_targets = {
                output_expr: 100.0,
                input_expr: 50.0,  # Inconsistent with 1:1 ratio
            }

            objective_weights = {
                output_expr: weight1,
                input_expr: weight2,
            }

            variable_bounds = {
                output_demand: (0.0, 200.0),
                input_supply: (0.0, 200.0),
                model.X[0]: (0.0, 200.0),
                model.Y[0]: (0.0, 200.0),
            }

            fixed_values = {
                model.S[0, 0]: 1.0,   # 1:1 input/output ratio
                model.U[1, 0]: 1.0,
            }

            milp_model = transformer.transform(
                objective_targets=objective_targets,
                objective_weights=objective_weights,
                variable_bounds=variable_bounds,
                fixed_values=fixed_values
            )

            backend = PythonMIPBackend()
            config = SolverConfig(verbose=False, time_limit=10.0)
            solution = backend.solve(milp_model, config)

            original_solution = transformer.extract_solution(solution.variables)
            Y_0 = original_solution.get(model.Y[0], 0.0)
            X_0 = original_solution.get(model.X[0], 0.0)

            return Y_0, X_0

        # Test 1: Equal weights (1.0, 1.0) - should be near middle
        output_equal, input_equal = solve_with_weights(1.0, 1.0)
        assert 60.0 < output_equal < 90.0, \
            f"With equal weights, should compromise around middle, got {output_equal}"

        # Test 2: Favor output (10.0, 1.0) - should be closer to 100
        output_high, input_high = solve_with_weights(10.0, 1.0)
        assert output_high > output_equal, \
            f"Higher output weight should increase output: {output_high} vs {output_equal}"
        assert output_high > 80.0, \
            f"With high output weight, should favor output target, got {output_high}"

        # Test 3: Favor input (1.0, 10.0) - should be closer to 50
        output_low, input_low = solve_with_weights(1.0, 10.0)
        assert output_low < output_equal, \
            f"Higher input weight should decrease output: {output_low} vs {output_equal}"
        assert output_low < 70.0, \
            f"With high input weight, should favor input target, got {output_low}"


class TestMILPNumericalStability:
    """Test numerical stability and edge cases."""

    def test_zero_target(self):
        """Test that zero targets work correctly."""
        processes = [
            Process(id="Process", produces=["Product"], consumes=[])
        ]
        objects = [
            Object(id="Product", metric=URIRef("http://qudt.org/vocab/quantitykind/Mass"))
        ]

        model = Model(processes=processes, objects=objects)

        amount = sy.Symbol("amount", nonnegative=True)
        model.add(model.pull_process_output("Process", "Product", amount))

        transformer = MILPTransformer(model)

        production_expr = model.expr("SoldProduction", object_id="Product")

        objective_targets = {
            production_expr: 0.0  # Zero target
        }

        variable_bounds = {
            amount: (0.0, 100.0),
            model.X[0]: (0.0, 100.0),
            model.Y[0]: (0.0, 100.0),
        }

        fixed_values = {
            model.S[0, 0]: 1.0,
        }

        milp_model = transformer.transform(
            objective_targets=objective_targets,
            variable_bounds=variable_bounds,
            fixed_values=fixed_values
        )

        backend = PythonMIPBackend()
        config = SolverConfig(verbose=False, time_limit=10.0)
        solution = backend.solve(milp_model, config)

        assert solution.status in ["optimal", "feasible"]

        original_solution = transformer.extract_solution(solution.variables)
        Y_0 = original_solution.get(model.Y[0], 0.0)

        assert Y_0 < 0.1, f"With zero target, production should be ~0, got {Y_0}"
        assert solution.objective_value < 0.1, \
            f"Objective should be near zero, got {solution.objective_value}"
