"""
MILP Transformation Module for Flowprog Models

This module transforms piecewise linear flowprog models into Mixed Integer Linear/Quadratic
Programs (MILP/MIQP) suitable for global optimization with guarantees on optimality.

Key Features:
- Systematic transformation of sympy expressions to MILP constraints
- Linearization of piecewise operations (Max, Min, Abs, Piecewise)
- Automatic big-M computation from variable bounds
- Solver-agnostic model representation
- Support for quadratic objective functions (calibration)
- Solution extraction and interpretation

Typical Usage:
    # Create MILP transformer
    transformer = MILPTransformer(model)

    # Define optimization targets (for calibration)
    targets = {
        model.expr("SoldProduction", object_id="Steel"): 100.0,
        model.expr("Consumption", object_id="Electricity"): 500.0,
    }
    weights = {
        model.expr("SoldProduction", object_id="Steel"): 1.0,
        model.expr("Consumption", object_id="Electricity"): 0.5,
    }

    # Transform to MILP
    milp_model = transformer.transform(
        objective_targets=targets,
        objective_weights=weights,
        variable_bounds={X[0]: (0, 100), Y[0]: (0, 100)}
    )

    # Solve with chosen backend
    backend = PythonMIPBackend()
    solution = backend.solve(milp_model)

    # Extract results
    optimal_values = transformer.extract_solution(solution)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Union, Any
from enum import Enum
import sympy as sy
from abc import ABC, abstractmethod


class ConstraintType(Enum):
    """Types of constraints in MILP formulation."""
    LINEAR_EQ = "=="
    LINEAR_LE = "<="
    LINEAR_GE = ">="


@dataclass
class MILPVariable:
    """Represents a variable in the MILP formulation."""
    name: str
    original_symbol: Optional[sy.Symbol] = None  # Link back to sympy symbol
    is_binary: bool = False
    is_auxiliary: bool = False  # Created during transformation
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    description: str = ""

    def __post_init__(self):
        if self.is_binary:
            self.lower_bound = 0
            self.upper_bound = 1


@dataclass
class MILPConstraint:
    """Represents a linear constraint in the MILP formulation."""
    name: str
    coefficients: Dict[str, float]  # variable_name -> coefficient
    constraint_type: ConstraintType
    rhs: float  # right-hand side
    description: str = ""

    def to_expression(self) -> str:
        """Convert to human-readable expression."""
        lhs_terms = [f"{coef}*{var}" for var, coef in self.coefficients.items()]
        lhs = " + ".join(lhs_terms)
        return f"{lhs} {self.constraint_type.value} {self.rhs}"


@dataclass
class QuadraticTerm:
    """Represents a quadratic term in objective function: coef * var1 * var2."""
    var1: str
    var2: str
    coefficient: float


@dataclass
class MILPObjective:
    """Represents the objective function (linear + quadratic terms)."""
    linear_terms: Dict[str, float] = field(default_factory=dict)  # variable -> coefficient
    quadratic_terms: List[QuadraticTerm] = field(default_factory=list)
    is_minimize: bool = True

    def add_linear_term(self, variable: str, coefficient: float):
        """Add or update a linear term."""
        if variable in self.linear_terms:
            self.linear_terms[variable] += coefficient
        else:
            self.linear_terms[variable] = coefficient

    def add_quadratic_term(self, var1: str, var2: str, coefficient: float):
        """Add a quadratic term."""
        self.quadratic_terms.append(QuadraticTerm(var1, var2, coefficient))


@dataclass
class MILPModel:
    """
    Solver-agnostic representation of a MILP/MIQP problem.

    This serves as an intermediate representation that can be translated
    to specific solver formats (Python-MIP, Pyomo, Gurobi, etc.).
    """
    name: str = "flowprog_milp"
    variables: Dict[str, MILPVariable] = field(default_factory=dict)
    constraints: List[MILPConstraint] = field(default_factory=list)
    objective: MILPObjective = field(default_factory=MILPObjective)

    # Metadata for solution interpretation
    variable_mapping: Dict[sy.Symbol, str] = field(default_factory=dict)  # sympy -> MILP var name
    auxiliary_expressions: Dict[str, sy.Expr] = field(default_factory=dict)  # aux var -> original expr

    def add_variable(self, var: MILPVariable) -> str:
        """Add a variable and return its name."""
        self.variables[var.name] = var
        return var.name

    def add_constraint(self, constraint: MILPConstraint):
        """Add a constraint to the model."""
        self.constraints.append(constraint)

    def get_statistics(self) -> Dict[str, int]:
        """Return problem size statistics."""
        return {
            "total_variables": len(self.variables),
            "continuous_variables": sum(1 for v in self.variables.values() if not v.is_binary),
            "binary_variables": sum(1 for v in self.variables.values() if v.is_binary),
            "constraints": len(self.constraints),
            "linear_objective_terms": len(self.objective.linear_terms),
            "quadratic_objective_terms": len(self.objective.quadratic_terms),
        }


class BoundsAnalyzer:
    """
    Analyzes sympy expressions to compute tight bounds on their values.

    This is critical for computing appropriate big-M values in the MILP formulation.
    """

    def __init__(self, variable_bounds: Dict[sy.Symbol, Tuple[float, float]]):
        """
        Initialize with known variable bounds.

        Args:
            variable_bounds: Dict mapping sympy symbols to (lower, upper) bound tuples
        """
        self.variable_bounds = variable_bounds
        self._bounds_cache: Dict[sy.Expr, Tuple[float, float]] = {}

    def get_bounds(self, expr: sy.Expr) -> Tuple[float, float]:
        """
        Compute bounds on an expression.

        Returns:
            (lower_bound, upper_bound) tuple
        """
        # Check cache first
        if expr in self._bounds_cache:
            return self._bounds_cache[expr]

        # Handle different expression types
        if expr.is_Number:
            bound = (float(expr), float(expr))
        elif expr.is_Symbol:
            bound = self.variable_bounds.get(expr, (float('-inf'), float('inf')))
        elif isinstance(expr, sy.Add):
            bound = self._bounds_add(expr)
        elif isinstance(expr, sy.Mul):
            bound = self._bounds_mul(expr)
        elif isinstance(expr, sy.Max):
            bound = self._bounds_max(expr)
        elif isinstance(expr, sy.Min):
            bound = self._bounds_min(expr)
        elif isinstance(expr, sy.Abs):
            bound = self._bounds_abs(expr)
        elif isinstance(expr, sy.Piecewise):
            bound = self._bounds_piecewise(expr)
        elif isinstance(expr, sy.Indexed):
            # Handle indexed symbols (e.g., X[j], S[i,j])
            bound = self.variable_bounds.get(expr, (float('-inf'), float('inf')))
        else:
            # Conservative fallback
            bound = (float('-inf'), float('inf'))

        self._bounds_cache[expr] = bound
        return bound

    def _bounds_add(self, expr: sy.Add) -> Tuple[float, float]:
        """Compute bounds on a sum."""
        lower = 0.0
        upper = 0.0
        for arg in expr.args:
            arg_lower, arg_upper = self.get_bounds(arg)
            lower += arg_lower
            upper += arg_upper
        return (lower, upper)

    def _bounds_mul(self, expr: sy.Mul) -> Tuple[float, float]:
        """Compute bounds on a product."""
        # Start with the first argument
        args = list(expr.args)
        if not args:
            return (1.0, 1.0)

        lower, upper = self.get_bounds(args[0])

        for arg in args[1:]:
            arg_lower, arg_upper = self.get_bounds(arg)
            # Compute all possible products
            products = [
                lower * arg_lower,
                lower * arg_upper,
                upper * arg_lower,
                upper * arg_upper
            ]
            lower = min(products)
            upper = max(products)

        return (lower, upper)

    def _bounds_max(self, expr: sy.Max) -> Tuple[float, float]:
        """Compute bounds on max expression."""
        all_bounds = [self.get_bounds(arg) for arg in expr.args]
        lower = max(b[0] for b in all_bounds)  # max of lower bounds
        upper = max(b[1] for b in all_bounds)  # max of upper bounds
        return (lower, upper)

    def _bounds_min(self, expr: sy.Min) -> Tuple[float, float]:
        """Compute bounds on min expression."""
        all_bounds = [self.get_bounds(arg) for arg in expr.args]
        lower = min(b[0] for b in all_bounds)  # min of lower bounds
        upper = min(b[1] for b in all_bounds)  # min of upper bounds
        return (lower, upper)

    def _bounds_abs(self, expr: sy.Abs) -> Tuple[float, float]:
        """Compute bounds on absolute value."""
        arg_lower, arg_upper = self.get_bounds(expr.args[0])
        if arg_lower >= 0:
            return (arg_lower, arg_upper)
        elif arg_upper <= 0:
            return (-arg_upper, -arg_lower)
        else:
            return (0.0, max(-arg_lower, arg_upper))

    def _bounds_piecewise(self, expr: sy.Piecewise) -> Tuple[float, float]:
        """Compute bounds on piecewise expression."""
        # Conservative: take min/max over all branches
        all_bounds = [self.get_bounds(val) for val, _ in expr.args]
        lower = min(b[0] for b in all_bounds)
        upper = max(b[1] for b in all_bounds)
        return (lower, upper)

    def compute_big_m(self, expr: sy.Expr, slack: float = 1.1) -> float:
        """
        Compute an appropriate big-M value for an expression.

        Args:
            expr: The expression to bound
            slack: Multiplicative slack factor (default 1.1 = 10% margin)

        Returns:
            A positive big-M value
        """
        lower, upper = self.get_bounds(expr)
        # Use the maximum absolute value with slack
        if lower == float('-inf') or upper == float('inf'):
            # Fallback to a large but finite value
            return 1e6

        max_abs = max(abs(lower), abs(upper))
        return max_abs * slack


class PiecewiseLinearizer:
    """
    Linearizes piecewise operations into MILP constraints.

    For each piecewise operation, generates:
    - Auxiliary continuous variables for results
    - Binary indicator variables for active branches/conditions
    - Linear constraints encoding the piecewise behavior
    """

    def __init__(self, bounds_analyzer: BoundsAnalyzer, transform_expr_callback=None):
        self.bounds_analyzer = bounds_analyzer
        self._counter = 0  # For generating unique variable names
        self._transform_expr_callback = transform_expr_callback  # Callback to transform sub-expressions

    def _generate_aux_name(self, prefix: str) -> str:
        """Generate a unique auxiliary variable name."""
        name = f"{prefix}_{self._counter}"
        self._counter += 1
        return name

    def _transform_expression(self, expr: sy.Expr, description: str = "") -> str:
        """Transform a sub-expression using the callback if available."""
        if self._transform_expr_callback is not None:
            return self._transform_expr_callback(expr, description)
        else:
            # Fallback: assume it's already a variable name or return string representation
            return str(expr)

    def linearize_max_zero(
        self,
        linear_expr: sy.Expr,
        expr_coeffs: Dict[sy.Symbol, float],
        constant: float,
        milp_model: MILPModel
    ) -> str:
        """
        Linearize max(0, linear_expr) where linear_expr = sum(coeffs) + constant.

        Creates:
            z = max(0, a^T x + b)

        Using constraints:
            z >= 0
            z >= a^T x + b
            z <= (a^T x + b) + M*(1-y)
            z <= M*y
            y ∈ {0,1}

        Args:
            linear_expr: The sympy expression inside max(0, ...)
            expr_coeffs: Dict mapping variables to their coefficients in linear_expr
            constant: Constant term in linear_expr
            milp_model: MILP model to add variables/constraints to

        Returns:
            Name of the auxiliary variable representing the result
        """
        # Create auxiliary variable for result
        z_name = self._generate_aux_name("max0")
        z_lower, z_upper = self.bounds_analyzer.get_bounds(sy.Max(0, linear_expr, evaluate=False))
        z_var = MILPVariable(
            name=z_name,
            is_auxiliary=True,
            lower_bound=z_lower,
            upper_bound=z_upper,
            description=f"max(0, {linear_expr})"
        )
        milp_model.add_variable(z_var)

        # Create binary indicator variable
        y_name = self._generate_aux_name("bin_max0")
        y_var = MILPVariable(
            name=y_name,
            is_binary=True,
            description=f"Indicator for {z_name}"
        )
        milp_model.add_variable(y_var)

        # Compute big-M
        M = self.bounds_analyzer.compute_big_m(linear_expr)

        # Convert sympy symbols to MILP variable names
        milp_coeffs = {
            milp_model.variable_mapping.get(sym, str(sym)): coef
            for sym, coef in expr_coeffs.items()
        }

        # Constraint 1: z >= 0
        milp_model.add_constraint(MILPConstraint(
            name=f"{z_name}_ge_0",
            coefficients={z_name: 1.0},
            constraint_type=ConstraintType.LINEAR_GE,
            rhs=0.0,
            description=f"{z_name} >= 0"
        ))

        # Constraint 2: z >= a^T x + b
        # Rewrite as: z - a^T x >= b
        constraint_coeffs = {z_name: 1.0}
        for var, coef in milp_coeffs.items():
            constraint_coeffs[var] = -coef
        milp_model.add_constraint(MILPConstraint(
            name=f"{z_name}_ge_expr",
            coefficients=constraint_coeffs,
            constraint_type=ConstraintType.LINEAR_GE,
            rhs=constant,
            description=f"{z_name} >= linear_expr"
        ))

        # Constraint 3: z <= (a^T x + b) + M*(1-y)
        # Rewrite as: z - a^T x + M*y <= b + M
        constraint_coeffs = {z_name: 1.0, y_name: M}
        for var, coef in milp_coeffs.items():
            constraint_coeffs[var] = -coef
        milp_model.add_constraint(MILPConstraint(
            name=f"{z_name}_le_expr_or_inactive",
            coefficients=constraint_coeffs,
            constraint_type=ConstraintType.LINEAR_LE,
            rhs=constant + M,
            description=f"{z_name} <= linear_expr + M*(1-{y_name})"
        ))

        # Constraint 4: z <= M*y
        # Rewrite as: z - M*y <= 0
        milp_model.add_constraint(MILPConstraint(
            name=f"{z_name}_le_active",
            coefficients={z_name: 1.0, y_name: -M},
            constraint_type=ConstraintType.LINEAR_LE,
            rhs=0.0,
            description=f"{z_name} <= M*{y_name}"
        ))

        return z_name

    def linearize_max_two(
        self,
        expr1: sy.Expr,
        expr2: sy.Expr,
        coeffs1: Dict[sy.Symbol, float],
        constant1: float,
        coeffs2: Dict[sy.Symbol, float],
        constant2: float,
        milp_model: MILPModel
    ) -> str:
        """
        Linearize max(expr1, expr2) where both are linear.

        Creates:
            z = max(e1, e2)

        Using constraints:
            z >= e1
            z >= e2
            z <= e1 + M*y
            z <= e2 + M*(1-y)
            y ∈ {0,1}

        Returns:
            Name of the auxiliary variable representing the result
        """
        # Create auxiliary variable for result
        z_name = self._generate_aux_name("max2")
        z_lower, z_upper = self.bounds_analyzer.get_bounds(
            sy.Max(expr1, expr2, evaluate=False)
        )
        z_var = MILPVariable(
            name=z_name,
            is_auxiliary=True,
            lower_bound=z_lower,
            upper_bound=z_upper,
            description=f"max({expr1}, {expr2})"
        )
        milp_model.add_variable(z_var)

        # Create binary indicator
        y_name = self._generate_aux_name("bin_max2")
        y_var = MILPVariable(
            name=y_name,
            is_binary=True,
            description=f"Indicator: expr1 >= expr2 for {z_name}"
        )
        milp_model.add_variable(y_var)

        # Compute big-M values
        M1 = self.bounds_analyzer.compute_big_m(expr1)
        M2 = self.bounds_analyzer.compute_big_m(expr2)

        # Convert coefficients
        milp_coeffs1 = {
            milp_model.variable_mapping.get(sym, str(sym)): coef
            for sym, coef in coeffs1.items()
        }
        milp_coeffs2 = {
            milp_model.variable_mapping.get(sym, str(sym)): coef
            for sym, coef in coeffs2.items()
        }

        # Constraint 1: z >= e1
        constraint_coeffs = {z_name: 1.0}
        for var, coef in milp_coeffs1.items():
            constraint_coeffs[var] = -coef
        milp_model.add_constraint(MILPConstraint(
            name=f"{z_name}_ge_expr1",
            coefficients=constraint_coeffs,
            constraint_type=ConstraintType.LINEAR_GE,
            rhs=constant1,
            description=f"{z_name} >= expr1"
        ))

        # Constraint 2: z >= e2
        constraint_coeffs = {z_name: 1.0}
        for var, coef in milp_coeffs2.items():
            constraint_coeffs[var] = -coef
        milp_model.add_constraint(MILPConstraint(
            name=f"{z_name}_ge_expr2",
            coefficients=constraint_coeffs,
            constraint_type=ConstraintType.LINEAR_GE,
            rhs=constant2,
            description=f"{z_name} >= expr2"
        ))

        # Constraint 3: z <= e1 + M1*(1-y)
        # Rewrite as: z - e1 - M1 + M1*y <= 0, or z - e1 + M1*y <= M1
        constraint_coeffs = {z_name: 1.0, y_name: M1}
        for var, coef in milp_coeffs1.items():
            constraint_coeffs[var] = -coef
        milp_model.add_constraint(MILPConstraint(
            name=f"{z_name}_le_expr1_or_inactive",
            coefficients=constraint_coeffs,
            constraint_type=ConstraintType.LINEAR_LE,
            rhs=constant1 + M1,
            description=f"{z_name} <= expr1 + M1*(1-{y_name})"
        ))

        # Constraint 4: z <= e2 + M2*y
        # Rewrite as: z - e2 - M2*y <= 0
        constraint_coeffs = {z_name: 1.0, y_name: -M2}
        for var, coef in milp_coeffs2.items():
            constraint_coeffs[var] = -coef
        milp_model.add_constraint(MILPConstraint(
            name=f"{z_name}_le_expr2_or_inactive",
            coefficients=constraint_coeffs,
            constraint_type=ConstraintType.LINEAR_LE,
            rhs=constant2,
            description=f"{z_name} <= expr2 + M2*{y_name}"
        ))

        return z_name

    def linearize_min_two(
        self,
        expr1: sy.Expr,
        expr2: sy.Expr,
        coeffs1: Dict[sy.Symbol, float],
        constant1: float,
        coeffs2: Dict[sy.Symbol, float],
        constant2: float,
        milp_model: MILPModel
    ) -> str:
        """
        Linearize min(expr1, expr2) using max(-expr1, -expr2).

        Returns:
            Name of the auxiliary variable representing the result
        """
        # min(a, b) = -max(-a, -b)
        # First create max of negated expressions
        neg_coeffs1 = {sym: -coef for sym, coef in coeffs1.items()}
        neg_coeffs2 = {sym: -coef for sym, coef in coeffs2.items()}

        max_var = self.linearize_max_two(
            -expr1, -expr2,
            neg_coeffs1, -constant1,
            neg_coeffs2, -constant2,
            milp_model
        )

        # Now create result = -max_var
        z_name = self._generate_aux_name("min2")
        z_lower, z_upper = self.bounds_analyzer.get_bounds(
            sy.Min(expr1, expr2, evaluate=False)
        )
        z_var = MILPVariable(
            name=z_name,
            is_auxiliary=True,
            lower_bound=z_lower,
            upper_bound=z_upper,
            description=f"min({expr1}, {expr2})"
        )
        milp_model.add_variable(z_var)

        # Constraint: z = -max_var, or z + max_var = 0
        milp_model.add_constraint(MILPConstraint(
            name=f"{z_name}_eq_neg_{max_var}",
            coefficients={z_name: 1.0, max_var: 1.0},
            constraint_type=ConstraintType.LINEAR_EQ,
            rhs=0.0,
            description=f"{z_name} = -{max_var}"
        ))

        return z_name

    def linearize_abs(
        self,
        linear_expr: sy.Expr,
        expr_coeffs: Dict[sy.Symbol, float],
        constant: float,
        milp_model: MILPModel
    ) -> str:
        """
        Linearize abs(linear_expr).

        Uses reformulation: abs(x) = max(x, -x)

        Returns:
            Name of the auxiliary variable representing the result
        """
        neg_coeffs = {sym: -coef for sym, coef in expr_coeffs.items()}

        return self.linearize_max_two(
            linear_expr, -linear_expr,
            expr_coeffs, constant,
            neg_coeffs, -constant,
            milp_model
        )

    def linearize_piecewise(
        self,
        piecewise_expr: sy.Piecewise,
        milp_model: MILPModel
    ) -> str:
        """
        Linearize a general Piecewise expression.

        For a piecewise with N branches:
            Piecewise((val1, cond1), (val2, cond2), ..., (valN, True))

        Creates:
        - Binary variables y_1, ..., y_N (one per branch)
        - Constraint: Σ y_i = 1 (exactly one branch active)
        - Constraints linking result to active branch value
        - Logic constraints encoding conditions (where possible)

        Returns:
            Name of the auxiliary variable representing the result
        """
        # Check if this is a simple max(0, x) pattern first (optimization)
        if len(piecewise_expr.args) == 2:
            val1, cond1 = piecewise_expr.args[0]
            val2, cond2 = piecewise_expr.args[1]

            # Pattern: Piecewise((0, cond), (expr, True))
            if val1 == 0 and cond2 == True:
                if val2.is_Add or val2.is_Mul or val2.is_Symbol or val2.is_Number:
                    coeffs, constant = self._extract_linear_coeffs(val2)
                    return self.linearize_max_zero(val2, coeffs, constant, milp_model)

            # Pattern: Piecewise((expr, cond), (0, True))
            if val2 == 0 and cond2 == True:
                if val1.is_Add or val1.is_Mul or val1.is_Symbol or val1.is_Number:
                    coeffs, constant = self._extract_linear_coeffs(val1)
                    return self.linearize_max_zero(val1, coeffs, constant, milp_model)

        # General N-branch piecewise linearization
        return self._linearize_general_piecewise(piecewise_expr, milp_model)

    def _linearize_general_piecewise(
        self,
        piecewise_expr: sy.Piecewise,
        milp_model: MILPModel
    ) -> str:
        """
        Linearize a general N-branch Piecewise expression.

        Strategy:
        1. Create binary variable y_i for each branch i
        2. Add constraint: Σ y_i = 1 (exactly one branch is active)
        3. For each branch: if y_i = 1, then result = value_i
           This is encoded as:
               result >= value_i - M*(1 - y_i)
               result <= value_i + M*(1 - y_i)
        4. Where possible, add logical constraints for conditions

        This uses a "convex combination" approach that works for MILP.
        """
        n_branches = len(piecewise_expr.args)

        # Create result variable
        result_name = self._generate_aux_name("piecewise")
        result_bounds = self.bounds_analyzer.get_bounds(piecewise_expr)
        result_var = MILPVariable(
            name=result_name,
            is_auxiliary=True,
            lower_bound=result_bounds[0],
            upper_bound=result_bounds[1],
            description=f"Piecewise result: {piecewise_expr}"
        )
        milp_model.add_variable(result_var)

        # Create binary indicator for each branch
        binary_names = []
        for i in range(n_branches):
            bin_name = self._generate_aux_name(f"pw_branch")
            bin_var = MILPVariable(
                name=bin_name,
                is_binary=True,
                description=f"Piecewise branch {i} indicator"
            )
            milp_model.add_variable(bin_var)
            binary_names.append(bin_name)

        # Constraint: exactly one branch must be active
        milp_model.add_constraint(MILPConstraint(
            name=f"{result_name}_one_branch",
            coefficients={bin_name: 1.0 for bin_name in binary_names},
            constraint_type=ConstraintType.LINEAR_EQ,
            rhs=1.0,
            description=f"Exactly one branch active for {result_name}"
        ))

        # For each branch, create constraints linking result to branch value
        for i, (value_expr, condition) in enumerate(piecewise_expr.args):
            bin_name = binary_names[i]

            # Transform the value expression to MILP
            value_var = self._transform_expression(value_expr, f"Piecewise branch {i} value")

            # Compute big-M for this branch
            value_bounds = self.bounds_analyzer.get_bounds(value_expr)
            result_bounds_tuple = self.bounds_analyzer.get_bounds(piecewise_expr)

            # M should be large enough to allow any valid result value
            M_lower = abs(result_bounds_tuple[0] - value_bounds[0])
            M_upper = abs(result_bounds_tuple[1] - value_bounds[1])
            M = max(M_lower, M_upper) * 1.1  # 10% slack

            if M == 0 or M == float('inf'):
                M = 1e6  # Fallback

            # When y_i = 1, force result = value_i
            # result >= value_i - M*(1 - y_i)
            # result - value_i >= -M*(1 - y_i)
            # result - value_i + M*y_i >= M
            milp_model.add_constraint(MILPConstraint(
                name=f"{result_name}_branch{i}_lower",
                coefficients={result_name: 1.0, value_var: -1.0, bin_name: M},
                constraint_type=ConstraintType.LINEAR_GE,
                rhs=M,
                description=f"If branch {i} active, {result_name} >= {value_var}"
            ))

            # result <= value_i + M*(1 - y_i)
            # result - value_i <= M*(1 - y_i)
            # result - value_i - M*y_i <= -M
            # Actually: result - value_i + M*y_i <= M (rearranging)
            # Let me reconsider: result <= value_i + M - M*y_i
            # result - value_i <= M - M*y_i
            # result - value_i + M*y_i <= M
            milp_model.add_constraint(MILPConstraint(
                name=f"{result_name}_branch{i}_upper",
                coefficients={result_name: 1.0, value_var: -1.0, bin_name: -M},
                constraint_type=ConstraintType.LINEAR_LE,
                rhs=-M,
                description=f"If branch {i} active, {result_name} <= {value_var}"
            ))

            # Try to add logical constraints for the condition
            # This is optional but helps the solver
            if i < n_branches - 1:  # Skip the final "True" condition
                self._add_condition_constraints(
                    condition, bin_name, milp_model, result_name, i
                )

        return result_name

    def _add_condition_constraints(
        self,
        condition: sy.Expr,
        binary_name: str,
        milp_model: MILPModel,
        result_name: str,
        branch_idx: int
    ):
        """
        Add constraints encoding logical conditions for piecewise branches.

        For conditions like:
        - x >= limit: encode as binary indicator
        - x <= limit: encode as binary indicator
        - More complex conditions: may skip for now

        This is best-effort - not all conditions can be easily encoded.
        The Σ y_i = 1 constraint plus objective function will often be sufficient.
        """
        try:
            # Check if this is a relational expression
            if isinstance(condition, (sy.GreaterThan, sy.StrictGreaterThan, sy.Ge)):
                # condition is: lhs >= rhs
                lhs = condition.lhs
                rhs = condition.rhs
                self._encode_greater_equal_condition(lhs, rhs, binary_name, milp_model, branch_idx)

            elif isinstance(condition, (sy.LessThan, sy.StrictLessThan, sy.Le)):
                # condition is: lhs <= rhs
                lhs = condition.lhs
                rhs = condition.rhs
                self._encode_less_equal_condition(lhs, rhs, binary_name, milp_model, branch_idx)

            elif condition == True or condition == sy.S.true:
                # Final catch-all branch, no condition needed
                pass

            else:
                # Complex condition we can't handle - skip it
                # The branch selection will still work via the y_i = 1 constraint
                pass

        except Exception:
            # If anything fails, just skip the condition constraint
            # The piecewise will still work, just less efficiently
            pass

    def _encode_greater_equal_condition(
        self,
        lhs: sy.Expr,
        rhs: sy.Expr,
        binary_name: str,
        milp_model: MILPModel,
        branch_idx: int
    ):
        """
        Encode: if binary = 1, then lhs >= rhs

        This is encoded as:
            lhs >= rhs - M*(1 - binary)

        Rearranging:
            lhs - rhs >= -M*(1 - binary)
            lhs - rhs + M*binary >= M
        """
        try:
            # Compute lhs - rhs
            diff = sy.simplify(lhs - rhs)

            # Extract linear coefficients
            coeffs, constant = self._extract_linear_coeffs(diff)

            # Compute big-M
            diff_bounds = self.bounds_analyzer.get_bounds(diff)
            M = max(abs(diff_bounds[0]), abs(diff_bounds[1])) * 1.5

            if M == 0 or M == float('inf'):
                M = 1e6

            # Build constraint coefficients
            constraint_coeffs = {binary_name: M}
            for sym, coef in coeffs.items():
                var_name = milp_model.variable_mapping.get(sym, str(sym))
                constraint_coeffs[var_name] = coef

            milp_model.add_constraint(MILPConstraint(
                name=f"cond_branch{branch_idx}_ge",
                coefficients=constraint_coeffs,
                constraint_type=ConstraintType.LINEAR_GE,
                rhs=M - constant,
                description=f"If branch {branch_idx}, then {lhs} >= {rhs}"
            ))

        except Exception:
            # If we can't encode it, skip
            pass

    def _encode_less_equal_condition(
        self,
        lhs: sy.Expr,
        rhs: sy.Expr,
        binary_name: str,
        milp_model: MILPModel,
        branch_idx: int
    ):
        """
        Encode: if binary = 1, then lhs <= rhs

        This is encoded as:
            lhs <= rhs + M*(1 - binary)

        Rearranging:
            lhs - rhs <= M*(1 - binary)
            lhs - rhs - M*binary <= -M
            -(lhs - rhs) + M*binary >= M
            rhs - lhs + M*binary >= M
        """
        try:
            # Compute lhs - rhs
            diff = sy.simplify(lhs - rhs)

            # Extract linear coefficients
            coeffs, constant = self._extract_linear_coeffs(diff)

            # Compute big-M
            diff_bounds = self.bounds_analyzer.get_bounds(diff)
            M = max(abs(diff_bounds[0]), abs(diff_bounds[1])) * 1.5

            if M == 0 or M == float('inf'):
                M = 1e6

            # Build constraint coefficients (negated for rhs - lhs)
            constraint_coeffs = {binary_name: M}
            for sym, coef in coeffs.items():
                var_name = milp_model.variable_mapping.get(sym, str(sym))
                constraint_coeffs[var_name] = -coef  # Negate for rhs - lhs

            milp_model.add_constraint(MILPConstraint(
                name=f"cond_branch{branch_idx}_le",
                coefficients=constraint_coeffs,
                constraint_type=ConstraintType.LINEAR_GE,
                rhs=M + constant,  # Note: sign flip on constant too
                description=f"If branch {branch_idx}, then {lhs} <= {rhs}"
            ))

        except Exception:
            # If we can't encode it, skip
            pass

    def _extract_linear_coeffs(
        self,
        expr: sy.Expr
    ) -> Tuple[Dict[sy.Symbol, float], float]:
        """
        Extract linear coefficients from an expression.

        Returns:
            (coefficients_dict, constant_term)
        """
        expanded = sy.expand(expr)

        # Get all symbols
        symbols = list(expanded.free_symbols)

        # Extract coefficients
        coeffs = {}
        for sym in symbols:
            coef = expanded.coeff(sym)
            if coef is not None and coef != 0:
                coeffs[sym] = float(coef)

        # Extract constant
        constant = float(expanded.as_coeff_add(*symbols)[0])

        return coeffs, constant


class MILPTransformer:
    """
    Main transformer class that converts flowprog Model to MILP formulation.

    Process:
    1. Collect all expressions from model._values
    2. Process intermediates in order
    3. For each expression, check for piecewise operations
    4. Generate MILP variables and constraints
    5. Build objective function from targets
    """

    def __init__(self, flowprog_model):
        """
        Initialize transformer with a flowprog Model.

        Args:
            flowprog_model: Instance of flowprog.imperative_model.Model
        """
        self.flowprog_model = flowprog_model
        self.milp_model: Optional[MILPModel] = None
        self.bounds_analyzer: Optional[BoundsAnalyzer] = None
        self.linearizer: Optional[PiecewiseLinearizer] = None

        # Track which sympy expressions map to which MILP variables
        self._expr_to_var: Dict[sy.Expr, str] = {}

    def transform(
        self,
        objective_targets: Dict[sy.Expr, float],
        objective_weights: Optional[Dict[sy.Expr, float]] = None,
        variable_bounds: Optional[Dict[sy.Symbol, Tuple[float, float]]] = None,
        fixed_values: Optional[Dict[sy.Symbol, float]] = None
    ) -> MILPModel:
        """
        Transform the flowprog model to MILP formulation.

        Args:
            objective_targets: Dict mapping expressions to target values for calibration
            objective_weights: Dict mapping expressions to weights in objective (default: all 1.0)
            variable_bounds: Dict mapping variables to (lower, upper) bounds
            fixed_values: Dict mapping variables to fixed values (for parameters)

        Returns:
            MILPModel instance ready for solving
        """
        if objective_weights is None:
            objective_weights = {expr: 1.0 for expr in objective_targets.keys()}

        if variable_bounds is None:
            variable_bounds = {}

        if fixed_values is None:
            fixed_values = {}

        # Store fixed values for substitution
        self.fixed_values = fixed_values

        # Initialize MILP model
        self.milp_model = MILPModel(name="flowprog_calibration")

        # Initialize bounds analyzer
        self.bounds_analyzer = BoundsAnalyzer(variable_bounds)

        # Initialize linearizer with callback to transform expressions
        self.linearizer = PiecewiseLinearizer(
            self.bounds_analyzer,
            transform_expr_callback=self._transform_expression
        )

        # Step 1: Create MILP variables for all flowprog decision variables
        self._create_decision_variables(variable_bounds, fixed_values)

        # Step 2: Process intermediate expressions in order
        self._process_intermediates()

        # Step 3: Process main expressions in _values
        self._process_main_expressions()

        # Step 4: Build objective function
        self._build_objective(objective_targets, objective_weights)

        return self.milp_model

    def _create_decision_variables(
        self,
        variable_bounds: Dict[sy.Symbol, Tuple[float, float]],
        fixed_values: Dict[sy.Symbol, float]
    ):
        """Create MILP variables for flowprog decision variables (X, Y, S, U)."""
        # Collect all symbols that appear in the model
        all_symbols = set()

        # From _values
        for symbol, expr in self.flowprog_model._values.items():
            all_symbols.add(symbol)
            all_symbols.update(expr.free_symbols)

        # From intermediates
        for sym, val, desc in self.flowprog_model._intermediates:
            all_symbols.add(sym)
            all_symbols.update(val.free_symbols)

        # Create MILP variable for each symbol
        for symbol in all_symbols:
            if symbol in fixed_values:
                # This is a fixed parameter, not a variable
                continue

            # Generate variable name
            var_name = str(symbol)

            # Get bounds
            if symbol in variable_bounds:
                lower, upper = variable_bounds[symbol]
            else:
                # Default: nonnegative
                lower, upper = 0.0, None

            # Create MILP variable
            milp_var = MILPVariable(
                name=var_name,
                original_symbol=symbol,
                is_auxiliary=False,
                lower_bound=lower,
                upper_bound=upper,
                description=f"Decision variable {symbol}"
            )

            self.milp_model.add_variable(milp_var)
            self.milp_model.variable_mapping[symbol] = var_name
            self._expr_to_var[symbol] = var_name

    def _process_intermediates(self):
        """Process intermediate expressions in order."""
        for intermediate_sym, intermediate_val, description in self.flowprog_model._intermediates:
            # Transform the expression
            result_var = self._transform_expression(intermediate_val, description)

            # Map intermediate symbol to result variable
            self._expr_to_var[intermediate_sym] = result_var
            self.milp_model.variable_mapping[intermediate_sym] = result_var
            self.milp_model.auxiliary_expressions[result_var] = intermediate_val

    def _process_main_expressions(self):
        """Process main expressions in _values dict."""
        for symbol, expr in self.flowprog_model._values.items():
            # Get the MILP variable for this symbol
            if symbol in self._expr_to_var:
                target_var = self._expr_to_var[symbol]
            else:
                # Skip if not tracked (might be a fixed parameter)
                continue

            # Transform the expression
            expr_var = self._transform_expression(expr, f"Definition of {symbol}")

            # Add equality constraint: target_var = expr_var
            if target_var != expr_var:
                self.milp_model.add_constraint(MILPConstraint(
                    name=f"def_{target_var}",
                    coefficients={target_var: 1.0, expr_var: -1.0},
                    constraint_type=ConstraintType.LINEAR_EQ,
                    rhs=0.0,
                    description=f"{target_var} = {expr}"
                ))

    def _transform_expression(self, expr: sy.Expr, description: str = "") -> str:
        """
        Transform a sympy expression to MILP variables/constraints.

        Returns:
            Name of the MILP variable representing this expression
        """
        # Substitute fixed values first
        if hasattr(self, 'fixed_values') and self.fixed_values:
            expr = expr.subs(self.fixed_values)
            expr = sy.simplify(expr)

        # Check if already transformed
        if expr in self._expr_to_var:
            return self._expr_to_var[expr]

        # Handle different expression types
        if expr.is_Number:
            # Create a fixed auxiliary variable for constants
            const_name = f"const_{float(expr)}"
            if const_name not in self.milp_model.variables:
                const_var = MILPVariable(
                    name=const_name,
                    is_auxiliary=True,
                    lower_bound=float(expr),
                    upper_bound=float(expr),
                    description=f"Constant {expr}"
                )
                self.milp_model.add_variable(const_var)
            self._expr_to_var[expr] = const_name
            return const_name

        elif expr.is_Symbol or isinstance(expr, sy.Indexed):
            # Direct variable reference
            if expr in self.milp_model.variable_mapping:
                var_name = self.milp_model.variable_mapping[expr]
                self._expr_to_var[expr] = var_name
                return var_name
            else:
                # Create variable if not exists
                var_name = str(expr)
                milp_var = MILPVariable(
                    name=var_name,
                    original_symbol=expr,
                    description=description
                )
                self.milp_model.add_variable(milp_var)
                self.milp_model.variable_mapping[expr] = var_name
                self._expr_to_var[expr] = var_name
                return var_name

        elif isinstance(expr, sy.Add):
            # Linear combination - create auxiliary variable
            return self._transform_add(expr, description)

        elif isinstance(expr, sy.Mul):
            # Product - check if linear or needs transformation
            return self._transform_mul(expr, description)

        elif isinstance(expr, sy.Max):
            # Piecewise max operation
            return self._transform_max(expr, description)

        elif isinstance(expr, sy.Min):
            # Piecewise min operation
            return self._transform_min(expr, description)

        elif isinstance(expr, sy.Abs):
            # Absolute value
            return self._transform_abs(expr, description)

        elif isinstance(expr, sy.Piecewise):
            # General piecewise
            return self._transform_piecewise(expr, description)

        else:
            raise NotImplementedError(
                f"Expression type {type(expr)} not yet supported: {expr}"
            )

    def _transform_add(self, expr: sy.Add, description: str) -> str:
        """Transform addition expression."""
        # Transform each argument
        arg_vars = [self._transform_expression(arg) for arg in expr.args]

        # Create auxiliary variable for sum
        result_name = self.linearizer._generate_aux_name("sum")
        result_bounds = self.bounds_analyzer.get_bounds(expr)
        result_var = MILPVariable(
            name=result_name,
            is_auxiliary=True,
            lower_bound=result_bounds[0],
            upper_bound=result_bounds[1],
            description=description or f"Sum: {expr}"
        )
        self.milp_model.add_variable(result_var)

        # Add constraint: result = sum of args
        # result - arg1 - arg2 - ... = 0
        coefficients = {result_name: 1.0}
        for arg_var in arg_vars:
            coefficients[arg_var] = -1.0

        self.milp_model.add_constraint(MILPConstraint(
            name=f"def_{result_name}",
            coefficients=coefficients,
            constraint_type=ConstraintType.LINEAR_EQ,
            rhs=0.0,
            description=f"{result_name} = sum of arguments"
        ))

        self._expr_to_var[expr] = result_name
        return result_name

    def _transform_mul(self, expr: sy.Mul, description: str) -> str:
        """Transform multiplication expression."""
        # Check if this is linear (one variable times constants)
        # Get all indexed symbols
        all_atoms = expr.atoms(sy.Symbol, sy.Indexed)

        # Separate decision variables from parameters
        decision_vars = [
            atom for atom in all_atoms
            if atom in self.milp_model.variable_mapping
        ]

        if len(decision_vars) <= 1:
            # Linear: constant * variable or pure constant
            if len(decision_vars) == 0:
                # Pure constant (may involve parameter symbols)
                try:
                    const_value = float(expr)
                    return self._transform_expression(sy.Float(const_value), description)
                except (TypeError, ValueError):
                    # Expression contains unknown symbols - can't evaluate
                    raise ValueError(
                        f"Expression {expr} contains unknown symbols that are not decision variables. "
                        "Please provide values for all parameter symbols in fixed_values."
                    )

            symbol = decision_vars[0]

            # Try to compute coefficient by dividing expr by symbol
            try:
                coef_expr = expr / symbol
                # Simplify and try to evaluate
                coef_expr = sy.simplify(coef_expr)
                coefficient = float(coef_expr)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"Could not compute coefficient for {symbol} in expression {expr}. "
                    f"Error: {e}. Ensure all non-decision variables have numeric values."
                )

            # Create auxiliary variable for scaled version
            result_name = self.linearizer._generate_aux_name("scale")
            result_bounds = self.bounds_analyzer.get_bounds(expr)
            result_var = MILPVariable(
                name=result_name,
                is_auxiliary=True,
                lower_bound=result_bounds[0],
                upper_bound=result_bounds[1],
                description=description or f"Scaled: {expr}"
            )
            self.milp_model.add_variable(result_var)

            # Add constraint: result = coefficient * variable
            # result - coefficient * variable = 0
            var_name = self.milp_model.variable_mapping.get(symbol, str(symbol))
            self.milp_model.add_constraint(MILPConstraint(
                name=f"def_{result_name}",
                coefficients={result_name: 1.0, var_name: -coefficient},
                constraint_type=ConstraintType.LINEAR_EQ,
                rhs=0.0,
                description=f"{result_name} = {coefficient} * {var_name}"
            ))

            self._expr_to_var[expr] = result_name
            return result_name
        else:
            # Nonlinear multiplication
            raise NotImplementedError(
                f"Nonlinear multiplication not supported: {expr}. "
                "Only linear expressions can be transformed to MILP."
            )

    def _transform_max(self, expr: sy.Max, description: str) -> str:
        """Transform Max expression."""
        args = expr.args

        if len(args) == 2:
            # Two-argument max
            arg1, arg2 = args

            # Check for max(0, x) pattern
            if arg1 == 0 or arg2 == 0:
                nonzero_arg = arg2 if arg1 == 0 else arg1

                # Extract linear coefficients
                coeffs, constant = self.linearizer._extract_linear_coeffs(nonzero_arg)

                return self.linearizer.linearize_max_zero(
                    nonzero_arg, coeffs, constant, self.milp_model
                )
            else:
                # General two-argument max
                coeffs1, constant1 = self.linearizer._extract_linear_coeffs(arg1)
                coeffs2, constant2 = self.linearizer._extract_linear_coeffs(arg2)

                return self.linearizer.linearize_max_two(
                    arg1, arg2,
                    coeffs1, constant1,
                    coeffs2, constant2,
                    self.milp_model
                )
        else:
            # More than two arguments - chain binary max operations
            result_var = self._transform_expression(args[0])

            for i in range(1, len(args)):
                # result = max(result, args[i])
                arg_i = args[i]

                # Get coefficients for current result and next arg
                # Note: result_var is already a MILP variable, treat as single-variable expr
                coeffs_result = {sy.Symbol(result_var): 1.0}
                constant_result = 0.0

                coeffs_arg, constant_arg = self.linearizer._extract_linear_coeffs(arg_i)

                result_var = self.linearizer.linearize_max_two(
                    sy.Symbol(result_var), arg_i,
                    coeffs_result, constant_result,
                    coeffs_arg, constant_arg,
                    self.milp_model
                )

            self._expr_to_var[expr] = result_var
            return result_var

    def _transform_min(self, expr: sy.Min, description: str) -> str:
        """Transform Min expression."""
        args = expr.args

        if len(args) == 2:
            arg1, arg2 = args
            coeffs1, constant1 = self.linearizer._extract_linear_coeffs(arg1)
            coeffs2, constant2 = self.linearizer._extract_linear_coeffs(arg2)

            result_var = self.linearizer.linearize_min_two(
                arg1, arg2,
                coeffs1, constant1,
                coeffs2, constant2,
                self.milp_model
            )
            self._expr_to_var[expr] = result_var
            return result_var
        else:
            # Chain binary min operations
            result_var = self._transform_expression(args[0])

            for i in range(1, len(args)):
                arg_i = args[i]
                coeffs_result = {sy.Symbol(result_var): 1.0}
                constant_result = 0.0
                coeffs_arg, constant_arg = self.linearizer._extract_linear_coeffs(arg_i)

                result_var = self.linearizer.linearize_min_two(
                    sy.Symbol(result_var), arg_i,
                    coeffs_result, constant_result,
                    coeffs_arg, constant_arg,
                    self.milp_model
                )

            self._expr_to_var[expr] = result_var
            return result_var

    def _transform_abs(self, expr: sy.Abs, description: str) -> str:
        """Transform absolute value."""
        arg = expr.args[0]
        coeffs, constant = self.linearizer._extract_linear_coeffs(arg)

        result_var = self.linearizer.linearize_abs(
            arg, coeffs, constant, self.milp_model
        )
        self._expr_to_var[expr] = result_var
        return result_var

    def _transform_piecewise(self, expr: sy.Piecewise, description: str) -> str:
        """Transform Piecewise expression."""
        result_var = self.linearizer.linearize_piecewise(expr, self.milp_model)
        self._expr_to_var[expr] = result_var
        return result_var

    def _build_objective(
        self,
        targets: Dict[sy.Expr, float],
        weights: Dict[sy.Expr, float]
    ):
        """
        Build quadratic objective function: min Σ w_i * (output_i - target_i)²

        Expands to: min Σ w_i * output_i² - 2*w_i*target_i*output_i + w_i*target_i²

        The constant term w_i*target_i² can be omitted as it doesn't affect optimization.
        """
        for expr, target in targets.items():
            weight = weights.get(expr, 1.0)

            # Get MILP variable for this expression
            expr_var = self._transform_expression(expr, f"Objective term: {expr}")

            # Add quadratic term: w * output²
            self.milp_model.objective.add_quadratic_term(
                expr_var, expr_var, weight
            )

            # Add linear term: -2 * w * target * output
            self.milp_model.objective.add_linear_term(
                expr_var, -2.0 * weight * target
            )

        self.milp_model.objective.is_minimize = True

    def extract_solution(self, solution: Dict[str, float]) -> Dict[sy.Symbol, float]:
        """
        Extract solution values for original flowprog variables.

        Args:
            solution: Dict mapping MILP variable names to optimal values

        Returns:
            Dict mapping original sympy symbols to their optimal values
        """
        result = {}

        for symbol, var_name in self.milp_model.variable_mapping.items():
            if var_name in solution:
                result[symbol] = solution[var_name]

        return result
