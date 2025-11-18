"""Analyzes and breaks down symbolic expressions for visualization."""

import sympy as sy
from typing import Any, Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class ExpressionNode:
    """Represents a node in the expression tree."""
    id: str
    label: str
    expression: str
    latex: str
    node_type: str  # 'operation', 'symbol', 'constant', 'intermediate'
    description: str = ""


@dataclass
class ExpressionEdge:
    """Represents a dependency edge in the expression tree."""
    source: str
    target: str


class ExpressionAnalyzer:
    """Analyzes symbolic expressions and breaks them down into explainable components."""

    def __init__(self, model):
        """Initialize analyzer with a flowprog Model instance."""
        self.model = model
        self._node_counter = 0

    def analyze_expression(self, expr: sy.Expr, name: str = "expression",
                          symbol_for_history: sy.Expr = None) -> Dict[str, Any]:
        """
        Analyze an expression and return its decomposition.

        Args:
            expr: The expression to analyze
            name: Display name for the expression
            symbol_for_history: Optional symbol to look up in history (e.g., X[0])
                               If not provided, uses expr

        Returns:
            Dictionary with:
            - 'final_expression': String representation
            - 'final_latex': LaTeX representation
            - 'nodes': List of ExpressionNode objects
            - 'edges': List of ExpressionEdge objects
            - 'intermediates': List of intermediate variable info
            - 'history': List of history labels
        """
        self._node_counter = 0
        nodes = []
        edges = []

        # Create root node
        root_id = self._next_id()
        nodes.append(ExpressionNode(
            id=root_id,
            label=name,
            expression=str(expr),
            latex=sy.latex(expr),
            node_type='root',
            description=f"Final expression for {name}"
        ))

        # Substitute intermediates and track them
        intermediates_info = self._get_intermediates_chain(expr)

        # Build expression tree
        self._build_expression_tree(expr, root_id, nodes, edges)

        # Get history if available
        history = []
        if hasattr(self.model, '_history'):
            # Use the provided symbol for history lookup, or fall back to expr
            lookup_symbol = symbol_for_history if symbol_for_history is not None else expr

            # Look up the symbol in history
            if lookup_symbol in self.model._history:
                history = self.model._history[lookup_symbol]

        return {
            'final_expression': str(expr),
            'final_latex': sy.latex(expr),
            'nodes': [self._node_to_dict(n) for n in nodes],
            'edges': [{'source': e.source, 'target': e.target} for e in edges],
            'intermediates': intermediates_info,
            'history': history
        }

    def _get_intermediates_chain(self, expr: sy.Expr) -> List[Dict[str, str]]:
        """Get the chain of intermediate variables involved in an expression."""
        intermediates = []

        if not hasattr(self.model, '_intermediates'):
            return intermediates

        # Find all intermediate symbols in the expression
        expr_symbols = expr.free_symbols

        # Build a map of intermediate symbols to their definitions
        intermediate_map = {}
        for sym, value, description in self.model._intermediates:
            intermediate_map[sym] = (value, description)

        # Find intermediates used in this expression
        seen = set()
        to_process = list(expr_symbols)

        while to_process:
            symbol = to_process.pop()
            if symbol in seen:
                continue
            seen.add(symbol)

            if symbol in intermediate_map:
                value, description = intermediate_map[symbol]
                intermediates.append({
                    'symbol': str(symbol),
                    'latex': sy.latex(symbol),
                    'value': str(value),
                    'value_latex': sy.latex(value),
                    'description': description
                })

                # Recursively find intermediates in the value
                to_process.extend(value.free_symbols)

        return intermediates

    def _build_expression_tree(self, expr: sy.Expr, parent_id: str,
                               nodes: List[ExpressionNode],
                               edges: List[ExpressionEdge],
                               max_depth: int = 3, current_depth: int = 0):
        """Recursively build the expression tree."""
        if current_depth >= max_depth:
            return

        # Handle different expression types
        if isinstance(expr, sy.Symbol):
            node_id = self._next_id()
            nodes.append(ExpressionNode(
                id=node_id,
                label=str(expr),
                expression=str(expr),
                latex=sy.latex(expr),
                node_type='symbol',
                description=self._describe_symbol(expr)
            ))
            edges.append(ExpressionEdge(source=node_id, target=parent_id))

        elif isinstance(expr, (sy.Integer, sy.Float, sy.Rational)):
            node_id = self._next_id()
            nodes.append(ExpressionNode(
                id=node_id,
                label=str(expr),
                expression=str(expr),
                latex=sy.latex(expr),
                node_type='constant'
            ))
            edges.append(ExpressionEdge(source=node_id, target=parent_id))

        elif hasattr(expr, 'args') and expr.args:
            # Operation node
            node_id = self._next_id()
            op_name = type(expr).__name__
            nodes.append(ExpressionNode(
                id=node_id,
                label=op_name,
                expression=str(expr),
                latex=sy.latex(expr),
                node_type='operation',
                description=self._describe_operation(expr)
            ))
            edges.append(ExpressionEdge(source=node_id, target=parent_id))

            # Process children
            for arg in expr.args:
                self._build_expression_tree(arg, node_id, nodes, edges,
                                           max_depth, current_depth + 1)

    def _describe_symbol(self, symbol: sy.Symbol) -> str:
        """Generate a description for a symbol."""
        name = str(symbol)

        # Check if it's a process input/output
        if hasattr(self.model, 'X') and isinstance(self.model.X, sy.IndexedBase):
            # Try to match X[i] or Y[i] patterns
            pass

        # Check if it's an intermediate
        if hasattr(self.model, '_intermediates'):
            for sym, value, description in self.model._intermediates:
                if sym == symbol:
                    return description

        # Check if it's a user parameter
        if name.startswith('Z_') or name.startswith('S_'):
            return f"User-defined parameter: {name}"

        return f"Symbol: {name}"

    def _describe_operation(self, expr: sy.Expr) -> str:
        """Generate a description for an operation."""
        op_type = type(expr).__name__

        if op_type == 'Add':
            return "Sum of terms"
        elif op_type == 'Mul':
            return "Product of factors"
        elif op_type == 'Pow':
            return "Exponentiation"
        elif op_type == 'Max':
            return "Maximum value"
        elif op_type == 'Min':
            return "Minimum value"
        elif op_type == 'Piecewise':
            return "Conditional expression (if-then-else logic)"
        else:
            return op_type

    def _symbols_match(self, sym1, sym2) -> bool:
        """Check if two symbols/expressions match."""
        return str(sym1) == str(sym2)

    def _node_to_dict(self, node: ExpressionNode) -> Dict[str, str]:
        """Convert ExpressionNode to dictionary."""
        return {
            'id': node.id,
            'label': node.label,
            'expression': node.expression,
            'latex': node.latex,
            'type': node.node_type,
            'description': node.description
        }

    def _next_id(self) -> str:
        """Generate next unique node ID."""
        node_id = f"node_{self._node_counter}"
        self._node_counter += 1
        return node_id
