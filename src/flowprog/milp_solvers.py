"""
Solver Backend Implementations for MILP Models

This module provides solver-specific implementations that can build and solve
MILPModel instances using various optimization libraries.

Supported backends:
- Python-MIP: Lightweight, supports CBC, Gurobi, CPLEX
- Direct dictionary export: For manual solver configuration
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, List, Tuple, Any
from dataclasses import dataclass

from flowprog.milp_transform import MILPModel, MILPVariable, MILPConstraint, ConstraintType


@dataclass
class SolverConfig:
    """Configuration options for MILP solvers."""
    time_limit: Optional[float] = None  # Maximum solve time in seconds
    mip_gap: Optional[float] = None  # Relative MIP gap tolerance
    threads: Optional[int] = None  # Number of threads to use
    verbose: bool = True  # Print solver output
    solver_name: Optional[str] = None  # Specific solver to use (backend-dependent)


@dataclass
class SolverSolution:
    """Solution returned by a solver."""
    status: str  # "optimal", "feasible", "infeasible", "unbounded", "error"
    objective_value: Optional[float] = None
    variables: Dict[str, float] = None  # variable name -> value
    gap: Optional[float] = None  # MIP gap (for non-optimal solutions)
    solve_time: Optional[float] = None
    num_nodes: Optional[int] = None  # Number of branch-and-bound nodes explored

    def __post_init__(self):
        if self.variables is None:
            self.variables = {}


class SolverBackend(ABC):
    """Abstract base class for solver backends."""

    @abstractmethod
    def solve(
        self,
        milp_model: MILPModel,
        config: Optional[SolverConfig] = None
    ) -> SolverSolution:
        """
        Build and solve a MILPModel.

        Args:
            milp_model: The MILP model to solve
            config: Solver configuration options

        Returns:
            SolverSolution with results
        """
        pass

    @abstractmethod
    def get_solver_name(self) -> str:
        """Return the name of this solver backend."""
        pass


class PythonMIPBackend(SolverBackend):
    """
    Backend using Python-MIP library.

    Python-MIP provides a unified interface to CBC (open-source),
    Gurobi, and CPLEX solvers.
    """

    def __init__(self):
        """Initialize Python-MIP backend."""
        try:
            import mip
            self.mip = mip
        except ImportError:
            raise ImportError(
                "Python-MIP library not found. Install with: pip install mip"
            )

    def get_solver_name(self) -> str:
        return "Python-MIP"

    def solve(
        self,
        milp_model: MILPModel,
        config: Optional[SolverConfig] = None
    ) -> SolverSolution:
        """Solve using Python-MIP."""
        if config is None:
            config = SolverConfig()

        # Create MIP model
        sense = self.mip.MINIMIZE if milp_model.objective.is_minimize else self.mip.MAXIMIZE

        # Select solver
        solver = self.mip.CBC  # Default to open-source CBC
        if config.solver_name:
            solver_map = {
                "cbc": self.mip.CBC,
                "gurobi": self.mip.GUROBI,
                "cplex": self.mip.CPLEX,
            }
            solver = solver_map.get(config.solver_name.lower(), self.mip.CBC)

        model = self.mip.Model(name=milp_model.name, sense=sense, solver_name=solver)

        # Set verbosity
        if not config.verbose:
            model.verbose = 0

        # Create variables
        mip_vars: Dict[str, Any] = {}  # MILP var name -> mip.Var

        for var_name, var_def in milp_model.variables.items():
            if var_def.is_binary:
                mip_var = model.add_var(
                    name=var_name,
                    var_type=self.mip.BINARY
                )
            else:
                var_type = self.mip.CONTINUOUS
                lb = var_def.lower_bound if var_def.lower_bound is not None else 0.0
                ub = var_def.upper_bound if var_def.upper_bound is not None else self.mip.INF

                mip_var = model.add_var(
                    name=var_name,
                    var_type=var_type,
                    lb=lb,
                    ub=ub
                )

            mip_vars[var_name] = mip_var

        # Add constraints
        for constraint in milp_model.constraints:
            # Build linear expression
            expr = self.mip.xsum(
                coef * mip_vars[var_name]
                for var_name, coef in constraint.coefficients.items()
            )

            # Add constraint based on type
            if constraint.constraint_type == ConstraintType.LINEAR_EQ:
                model.add_constr(expr == constraint.rhs, name=constraint.name)
            elif constraint.constraint_type == ConstraintType.LINEAR_LE:
                model.add_constr(expr <= constraint.rhs, name=constraint.name)
            elif constraint.constraint_type == ConstraintType.LINEAR_GE:
                model.add_constr(expr >= constraint.rhs, name=constraint.name)

        # Build objective function
        obj_expr = 0

        # Check if we have quadratic terms and if solver supports them
        if milp_model.objective.quadratic_terms:
            solver = config.solver_name.lower() if config.solver_name else "cbc"

            if solver in ["gurobi", "cplex"]:
                # These solvers support native quadratic objectives
                # Add linear terms first
                if milp_model.objective.linear_terms:
                    obj_expr += self.mip.xsum(
                        coef * mip_vars[var_name]
                        for var_name, coef in milp_model.objective.linear_terms.items()
                    )

                # Add quadratic terms
                for quad_term in milp_model.objective.quadratic_terms:
                    var1 = mip_vars[quad_term.var1]
                    var2 = mip_vars[quad_term.var2]
                    coef = quad_term.coefficient

                    # Quadratic term: coef * var1 * var2
                    obj_expr += coef * var1 * var2

            else:
                # CBC doesn't support quadratic - need to linearize or approximate
                # For sum of squares objective: Σ w_i * x_i²
                # We can use L1 approximation instead: Σ w_i * |x_i|
                # Or just use linear objective with absolute values

                import warnings
                warnings.warn(
                    f"Solver '{solver}' does not support quadratic objectives. "
                    "Falling back to linear approximation (L1 norm instead of L2). "
                    "For better results, install Gurobi or CPLEX.",
                    UserWarning
                )

                # For L1 approximation of (x - target)²:
                # The objective is: Σ w_i * x_i² - 2*w_i*target_i*x_i + constant
                # We need to reconstruct (x - target) and minimize |x - target|

                # Group quadratic and linear terms by variable
                error_terms = {}  # var_name -> (weight, target)

                for quad_term in milp_model.objective.quadratic_terms:
                    if quad_term.var1 == quad_term.var2:
                        var_name = quad_term.var1
                        weight = quad_term.coefficient

                        # Find corresponding linear term to extract target
                        linear_coef = milp_model.objective.linear_terms.get(var_name, 0.0)
                        # linear_coef = -2 * weight * target
                        # So: target = -linear_coef / (2 * weight)
                        if weight != 0:
                            target = -linear_coef / (2.0 * weight)
                            error_terms[var_name] = (weight, target)

                # Create auxiliary variables for |x - target|
                for var_name, (weight, target) in error_terms.items():
                    var = mip_vars[var_name]

                    # Create auxiliary variable for error = x - target
                    error_var_name = f"error_{var_name}"
                    error_var = model.add_var(
                        name=error_var_name,
                        var_type=self.mip.CONTINUOUS,
                        lb=-self.mip.INF
                    )

                    # error = x - target
                    model.add_constr(error_var == var - target, name=f"def_{error_var_name}")

                    # Create auxiliary variable for |error|
                    abs_error_var_name = f"abs_error_{var_name}"
                    abs_error_var = model.add_var(
                        name=abs_error_var_name,
                        var_type=self.mip.CONTINUOUS,
                        lb=0
                    )

                    # |error| constraints
                    model.add_constr(abs_error_var >= error_var, name=f"{abs_error_var_name}_pos")
                    model.add_constr(abs_error_var >= -error_var, name=f"{abs_error_var_name}_neg")

                    # Add to objective: weight * |error|
                    obj_expr += weight * abs_error_var

                # Clear the quadratic and linear terms that we've handled
                # (they've been replaced with L1 approximation)
                handled_vars = set(error_terms.keys())
                remaining_linear = {
                    var_name: coef
                    for var_name, coef in milp_model.objective.linear_terms.items()
                    if var_name not in handled_vars
                }

                # Add any remaining linear terms
                if remaining_linear:
                    obj_expr += self.mip.xsum(
                        coef * mip_vars[var_name]
                        for var_name, coef in remaining_linear.items()
                    )
        else:
            # No quadratic terms - just use linear objective
            if milp_model.objective.linear_terms:
                obj_expr += self.mip.xsum(
                    coef * mip_vars[var_name]
                    for var_name, coef in milp_model.objective.linear_terms.items()
                )

        model.objective = obj_expr

        # Set solver parameters
        if config.time_limit:
            model.max_seconds = config.time_limit

        if config.mip_gap:
            model.max_mip_gap = config.mip_gap

        if config.threads:
            model.threads = config.threads

        # Solve
        status = model.optimize()

        # Extract solution
        status_map = {
            self.mip.OptimizationStatus.OPTIMAL: "optimal",
            self.mip.OptimizationStatus.FEASIBLE: "feasible",
            self.mip.OptimizationStatus.INFEASIBLE: "infeasible",
            self.mip.OptimizationStatus.UNBOUNDED: "unbounded",
            self.mip.OptimizationStatus.ERROR: "error",
            self.mip.OptimizationStatus.NO_SOLUTION_FOUND: "no_solution",
        }

        solution_status = status_map.get(status, "unknown")

        if status in [self.mip.OptimizationStatus.OPTIMAL, self.mip.OptimizationStatus.FEASIBLE]:
            # Extract variable values
            solution_vars = {
                var_name: mip_var.x
                for var_name, mip_var in mip_vars.items()
                if mip_var.x is not None
            }

            return SolverSolution(
                status=solution_status,
                objective_value=model.objective_value,
                variables=solution_vars,
                gap=model.gap if hasattr(model, 'gap') else None,
                solve_time=None,  # Python-MIP doesn't expose solve time easily
                num_nodes=model.num_solutions if hasattr(model, 'num_solutions') else None
            )
        else:
            return SolverSolution(
                status=solution_status,
                solve_time=None
            )

    def find_multiple_solutions(
        self,
        milp_model: MILPModel,
        num_solutions: int,
        tolerance: Optional[float] = None,
        config: Optional[SolverConfig] = None
    ) -> List[SolverSolution]:
        """
        Find multiple near-optimal solutions.

        Args:
            milp_model: The MILP model to solve
            num_solutions: Maximum number of solutions to find
            tolerance: Objective value tolerance (solutions within this from optimal)
            config: Solver configuration

        Returns:
            List of solutions
        """
        if config is None:
            config = SolverConfig()

        # First, find optimal solution
        optimal_solution = self.solve(milp_model, config)

        if optimal_solution.status != "optimal":
            return [optimal_solution]

        solutions = [optimal_solution]

        if tolerance is None:
            tolerance = 0.01  # 1% default

        optimal_value = optimal_solution.objective_value

        # Create model with solution pool
        sense = self.mip.MINIMIZE if milp_model.objective.is_minimize else self.mip.MAXIMIZE
        model = self.mip.Model(name=milp_model.name, sense=sense)

        if not config.verbose:
            model.verbose = 0

        # Build model (same as solve())
        mip_vars = {}
        for var_name, var_def in milp_model.variables.items():
            if var_def.is_binary:
                mip_var = model.add_var(name=var_name, var_type=self.mip.BINARY)
            else:
                lb = var_def.lower_bound if var_def.lower_bound is not None else 0.0
                ub = var_def.upper_bound if var_def.upper_bound is not None else self.mip.INF
                mip_var = model.add_var(name=var_name, var_type=self.mip.CONTINUOUS, lb=lb, ub=ub)
            mip_vars[var_name] = mip_var

        for constraint in milp_model.constraints:
            expr = self.mip.xsum(
                coef * mip_vars[var_name]
                for var_name, coef in constraint.coefficients.items()
            )
            if constraint.constraint_type == ConstraintType.LINEAR_EQ:
                model.add_constr(expr == constraint.rhs, name=constraint.name)
            elif constraint.constraint_type == ConstraintType.LINEAR_LE:
                model.add_constr(expr <= constraint.rhs, name=constraint.name)
            elif constraint.constraint_type == ConstraintType.LINEAR_GE:
                model.add_constr(expr >= constraint.rhs, name=constraint.name)

        obj_expr = 0
        if milp_model.objective.linear_terms:
            obj_expr += self.mip.xsum(
                coef * mip_vars[var_name]
                for var_name, coef in milp_model.objective.linear_terms.items()
            )
        if milp_model.objective.quadratic_terms:
            for quad_term in milp_model.objective.quadratic_terms:
                obj_expr += quad_term.coefficient * mip_vars[quad_term.var1] * mip_vars[quad_term.var2]

        model.objective = obj_expr

        # Add constraint to enforce solution quality
        if milp_model.objective.is_minimize:
            model.add_constr(obj_expr <= optimal_value * (1.0 + tolerance))
        else:
            model.add_constr(obj_expr >= optimal_value * (1.0 - tolerance))

        # Try to find additional solutions by adding exclusion constraints
        for _ in range(num_solutions - 1):
            # Solve again
            status = model.optimize()

            if status not in [self.mip.OptimizationStatus.OPTIMAL, self.mip.OptimizationStatus.FEASIBLE]:
                break

            # Extract solution
            solution_vars = {
                var_name: mip_var.x
                for var_name, mip_var in mip_vars.items()
                if mip_var.x is not None
            }

            solutions.append(SolverSolution(
                status="feasible",
                objective_value=model.objective_value,
                variables=solution_vars,
                gap=model.gap,
                solve_time=model.total_time
            ))

            # Add exclusion constraint based on binary variables
            binary_vars = [
                (var_name, mip_var)
                for var_name, mip_var in mip_vars.items()
                if milp_model.variables[var_name].is_binary
            ]

            if not binary_vars:
                # No binary variables to exclude on
                break

            # Exclude current binary solution
            # Sum of different binaries >= 1
            exclusion_expr = self.mip.xsum(
                mip_var if mip_var.x < 0.5 else (1 - mip_var)
                for _, mip_var in binary_vars
            )
            model.add_constr(exclusion_expr >= 1, name=f"exclude_sol_{len(solutions)}")

        return solutions


class DictExportBackend(SolverBackend):
    """
    Backend that exports the MILP model as nested dictionaries.

    Useful for manual inspection or integration with custom solvers.
    """

    def get_solver_name(self) -> str:
        return "Dictionary Export"

    def solve(
        self,
        milp_model: MILPModel,
        config: Optional[SolverConfig] = None
    ) -> SolverSolution:
        """
        Export model to dictionary format.

        Returns a SolverSolution with status="exported" and the model
        structure in a special format.
        """
        model_dict = self.export_to_dict(milp_model)

        # Return special solution indicating export
        return SolverSolution(
            status="exported",
            variables={"_model_dict": model_dict}  # Store dict in special key
        )

    def export_to_dict(self, milp_model: MILPModel) -> Dict[str, Any]:
        """
        Export MILPModel to nested dictionary structure.

        Returns:
            {
                "name": str,
                "variables": [
                    {
                        "name": str,
                        "type": "continuous" | "binary",
                        "lower_bound": float,
                        "upper_bound": float,
                        "description": str
                    },
                    ...
                ],
                "constraints": [
                    {
                        "name": str,
                        "type": "==" | "<=" | ">=",
                        "coefficients": {"var1": coef1, ...},
                        "rhs": float,
                        "description": str
                    },
                    ...
                ],
                "objective": {
                    "sense": "minimize" | "maximize",
                    "linear_terms": {"var1": coef1, ...},
                    "quadratic_terms": [
                        {"var1": str, "var2": str, "coefficient": float},
                        ...
                    ]
                },
                "statistics": {...}
            }
        """
        variables_list = []
        for var_name, var_def in milp_model.variables.items():
            variables_list.append({
                "name": var_name,
                "type": "binary" if var_def.is_binary else "continuous",
                "lower_bound": var_def.lower_bound,
                "upper_bound": var_def.upper_bound,
                "is_auxiliary": var_def.is_auxiliary,
                "description": var_def.description
            })

        constraints_list = []
        for constraint in milp_model.constraints:
            constraints_list.append({
                "name": constraint.name,
                "type": constraint.constraint_type.value,
                "coefficients": constraint.coefficients,
                "rhs": constraint.rhs,
                "description": constraint.description,
                "expression": constraint.to_expression()
            })

        quadratic_terms_list = []
        for quad_term in milp_model.objective.quadratic_terms:
            quadratic_terms_list.append({
                "var1": quad_term.var1,
                "var2": quad_term.var2,
                "coefficient": quad_term.coefficient
            })

        return {
            "name": milp_model.name,
            "variables": variables_list,
            "constraints": constraints_list,
            "objective": {
                "sense": "minimize" if milp_model.objective.is_minimize else "maximize",
                "linear_terms": milp_model.objective.linear_terms,
                "quadratic_terms": quadratic_terms_list
            },
            "statistics": milp_model.get_statistics()
        }


def get_available_solvers() -> Dict[str, bool]:
    """
    Check which solver backends are available.

    Returns:
        Dict mapping solver names to availability status
    """
    solvers = {}

    # Check Python-MIP
    try:
        import mip
        solvers["python-mip"] = True
    except ImportError:
        solvers["python-mip"] = False

    # Dict export is always available
    solvers["dict-export"] = True

    return solvers


def get_recommended_backend() -> SolverBackend:
    """
    Get the recommended solver backend based on availability.

    Returns:
        An instance of the best available solver backend
    """
    available = get_available_solvers()

    if available["python-mip"]:
        return PythonMIPBackend()
    else:
        return DictExportBackend()
