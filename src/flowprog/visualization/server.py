"""Flask server for interactive model visualization."""

import json
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import sympy as sy
from typing import Dict, List, Any

from .expression_analyzer import ExpressionAnalyzer


class VisualizationServer:
    """Flask server for visualizing flowprog models."""

    def __init__(self, model, recipe_data: Dict = None, parameter_values: Dict = None):
        """
        Initialize visualization server.

        Args:
            model: flowprog Model instance
            recipe_data: Dictionary of recipe coefficients (S and U values)
            parameter_values: Dictionary of parameter values for evaluation
        """
        self.model = model
        self.recipe_data = recipe_data or {}
        self.parameter_values = parameter_values or {}
        self.analyzer = ExpressionAnalyzer(model)

        # Create Flask app
        self.app = Flask(__name__,
                        template_folder='templates',
                        static_folder='static')
        CORS(self.app)

        # Register routes
        self._register_routes()

    def _register_routes(self):
        """Register all Flask routes."""

        @self.app.route('/')
        def index():
            return render_template('index.html')

        @self.app.route('/api/graph')
        def get_graph():
            """Get the complete graph structure (nodes and edges)."""
            return jsonify(self._build_graph_data())

        @self.app.route('/api/process/<process_id>')
        def get_process_details(process_id):
            """Get detailed information about a process."""
            return jsonify(self._get_process_details(process_id))

        @self.app.route('/api/flow/<source>/<target>/<material>')
        def get_flow_details(source, target, material):
            """Get detailed information about a flow."""
            return jsonify(self._get_flow_details(source, target, material))

        @self.app.route('/api/parameters', methods=['GET', 'POST'])
        def parameters():
            """Get or update parameter values."""
            if request.method == 'POST':
                self.parameter_values.update(request.json)
                return jsonify({'status': 'ok', 'parameters': self.parameter_values})
            else:
                return jsonify(self.parameter_values)

    def _build_graph_data(self) -> Dict[str, Any]:
        """Build the graph structure with nodes and edges."""
        nodes = []
        edges = []

        # Add process nodes
        for idx, process in enumerate(self.model.processes):
            nodes.append({
                'data': {
                    'id': process.id,
                    'label': process.id,
                    'type': 'process',
                    'has_stock': process.has_stock,
                    'produces': process.produces,
                    'consumes': process.consumes
                },
                'classes': 'process'
            })

        # Add object nodes
        for idx, obj in enumerate(self.model.objects):
            nodes.append({
                'data': {
                    'id': obj.id,
                    'label': obj.id,
                    'type': 'object',
                    'metric': str(obj.metric),
                    'has_market': obj.has_market
                },
                'classes': 'object market' if obj.has_market else 'object'
            })

        # Add flow edges (both production and consumption)
        # Get flows from the model
        try:
            flows_df = self.model.to_flows(self.recipe_data)

            for _, row in flows_df.iterrows():
                source = row['source']
                target = row['target']
                material = row['material']
                value = row['value']

                # Evaluate value if possible
                display_value = self._evaluate_expression(value)

                edge_id = f"{source}_{target}_{material}"

                # Don't show long expressions on edges, just the material name
                # Numeric values can be shown if short
                edge_label = material
                if display_value and len(display_value) < 15:
                    try:
                        # Only add value if it's numeric
                        float(display_value)
                        edge_label = f"{material}\n{display_value}"
                    except (ValueError, TypeError):
                        # Symbolic expression, don't add to label
                        pass

                edges.append({
                    'data': {
                        'id': edge_id,
                        'source': source,
                        'target': target,
                        'label': edge_label,
                        'material': material,
                        'value': str(value),
                        'evaluated_value': display_value
                    },
                    'classes': 'flow'
                })
        except Exception as e:
            print(f"Warning: Could not generate flows: {e}")

        return {
            'nodes': nodes,
            'edges': edges
        }

    def _get_process_details(self, process_id: str) -> Dict[str, Any]:
        """Get detailed information about a process including X and Y expressions."""
        try:
            # Find process index
            proc_idx = self.model._process_name_to_idx.get(process_id)
            if proc_idx is None:
                return {'error': f'Process {process_id} not found'}

            process = self.model.processes[proc_idx]

            # Get X and Y values
            x_symbol = self.model.X[proc_idx]
            y_symbol = self.model.Y[proc_idx]
            x_expr = self.model._values.get(x_symbol, sy.Integer(0))
            y_expr = self.model._values.get(y_symbol, sy.Integer(0))

            # Analyze expressions, passing the symbol for history lookup
            x_analysis = self.analyzer.analyze_expression(
                x_expr, f"X[{process_id}] (Process Input)",
                symbol_for_history=x_symbol
            )
            y_analysis = self.analyzer.analyze_expression(
                y_expr, f"Y[{process_id}] (Process Output)",
                symbol_for_history=y_symbol
            )

            # Add evaluation modes
            x_analysis['evaluation_modes'] = self._get_expression_with_modes(x_expr)
            y_analysis['evaluation_modes'] = self._get_expression_with_modes(y_expr)

            # Get input and output flows
            inputs = []
            outputs = []

            for obj_id in process.consumes:
                obj_idx = self.model._obj_name_to_idx[obj_id]
                # Get U coefficient from recipe_data, not _values
                u_coeff = self.recipe_data.get(self.model.U[obj_idx, proc_idx], sy.Integer(0))
                flow_expr = x_expr * u_coeff
                inputs.append({
                    'object': obj_id,
                    'expression': str(flow_expr),
                    'latex': sy.latex(flow_expr),
                    'value': self._evaluate_expression(flow_expr)
                })

            for obj_id in process.produces:
                obj_idx = self.model._obj_name_to_idx[obj_id]
                # Get S coefficient from recipe_data, not _values
                s_coeff = self.recipe_data.get(self.model.S[obj_idx, proc_idx], sy.Integer(0))
                flow_expr = y_expr * s_coeff
                outputs.append({
                    'object': obj_id,
                    'expression': str(flow_expr),
                    'latex': sy.latex(flow_expr),
                    'value': self._evaluate_expression(flow_expr)
                })

            return {
                'process_id': process_id,
                'process_index': proc_idx,
                'has_stock': process.has_stock,
                'consumes': process.consumes,
                'produces': process.produces,
                'x_analysis': x_analysis,
                'y_analysis': y_analysis,
                'inputs': inputs,
                'outputs': outputs
            }

        except Exception as e:
            return {'error': str(e)}

    def _get_flow_details(self, source: str, target: str, material: str) -> Dict[str, Any]:
        """Get detailed information about a flow."""
        try:
            # Determine if this is a production or consumption flow
            # Production: process -> object
            # Consumption: object -> process

            is_production = source in self.model._process_name_to_idx

            if is_production:
                process_id = source
                object_id = target
                proc_idx = self.model._process_name_to_idx[process_id]
                obj_idx = self.model._obj_name_to_idx[object_id]

                # Flow = Y[j] * S[i, j]
                y_expr = self.model._values.get(self.model.Y[proc_idx], sy.Integer(0))
                # Get S coefficient from recipe_data, not _values
                s_coeff = self.recipe_data.get(self.model.S[obj_idx, proc_idx], sy.Integer(0))
                flow_expr = y_expr * s_coeff

                flow_type = "Production"
                description = f"Production of {material} from process {process_id}"

            else:
                object_id = source
                process_id = target
                proc_idx = self.model._process_name_to_idx[process_id]
                obj_idx = self.model._obj_name_to_idx[object_id]

                # Flow = X[j] * U[i, j]
                x_expr = self.model._values.get(self.model.X[proc_idx], sy.Integer(0))
                # Get U coefficient from recipe_data, not _values
                u_coeff = self.recipe_data.get(self.model.U[obj_idx, proc_idx], sy.Integer(0))
                flow_expr = x_expr * u_coeff

                flow_type = "Consumption"
                description = f"Consumption of {material} by process {process_id}"

            # Analyze the flow expression
            analysis = self.analyzer.analyze_expression(flow_expr, description)

            # Add evaluation modes
            analysis['evaluation_modes'] = self._get_expression_with_modes(flow_expr)

            return {
                'source': source,
                'target': target,
                'material': material,
                'flow_type': flow_type,
                'description': description,
                'analysis': analysis,
                'evaluated_value': self._evaluate_expression(flow_expr)
            }

        except Exception as e:
            return {'error': str(e)}

    def _evaluate_expression(self, expr, mode='full') -> str:
        """
        Evaluate an expression with current parameter values.

        Args:
            expr: Symbolic expression or numeric value
            mode: Evaluation mode - 'symbolic' (no substitution),
                  'recipe' (substitute S/U only), or 'full' (substitute all)

        Returns:
            String representation of the expression
        """
        if isinstance(expr, (int, float)):
            return f"{expr:.4g}"

        if not isinstance(expr, sy.Expr):
            return str(expr)

        try:
            if mode == 'symbolic':
                # No substitution, return as-is
                return str(expr)

            elif mode == 'recipe':
                # Only substitute recipe coefficients (S and U)
                result = expr.subs(self.recipe_data)
                if result.is_number:
                    return f"{float(result):.4g}"
                return str(result)

            else:  # mode == 'full'
                # First substitute intermediates, then substitute all parameters
                all_params = {**self.recipe_data, **self.parameter_values}

                # Use model's eval_intermediates to substitute intermediate symbols (x0, x1, etc.)
                if hasattr(self.model, 'eval_intermediates'):
                    # eval_intermediates substitutes intermediates but doesn't substitute S/U/parameters
                    # So we need to do both steps
                    result = self.model.eval_intermediates(expr, all_params)
                    # Now substitute recipe and parameters into the result
                    result = result.subs(all_params)
                else:
                    result = expr.subs(all_params)

                if result.is_number:
                    return f"{float(result):.4g}"
                return str(result)

        except Exception:
            return str(expr)

    def _get_expression_with_modes(self, expr) -> Dict[str, str]:
        """Get an expression evaluated in all three modes, with validation."""
        recipe_evaluated = self._evaluate_expression(expr, mode='recipe')

        # Check if recipe-evaluated expression still contains S or U symbols
        # (indicates missing recipe coefficients)
        missing_coefficients = []
        try:
            # Re-substitute to get the sympy expression
            result = expr.subs(self.recipe_data)

            # Check for S and U symbols in the result
            for symbol in result.free_symbols:
                symbol_str = str(symbol)
                # Check if it's an indexed S or U (like S[3,2] or U[0,3])
                if symbol_str.startswith('S[') or symbol_str.startswith('U['):
                    missing_coefficients.append(symbol_str)
        except:
            pass

        # Generate LaTeX for each mode
        # Symbolic: original expression
        symbolic_latex = sy.latex(expr)

        # Recipe evaluated: substitute recipe coefficients
        try:
            recipe_expr = expr.subs(self.recipe_data)
            recipe_latex = sy.latex(recipe_expr)
        except:
            recipe_latex = symbolic_latex

        # Fully evaluated: substitute all parameters and intermediates
        try:
            all_params = {**self.recipe_data, **self.parameter_values}
            if hasattr(self.model, 'eval_intermediates'):
                fully_expr = self.model.eval_intermediates(expr, all_params)
                fully_expr = fully_expr.subs(all_params)
            else:
                fully_expr = expr.subs(all_params)
            fully_latex = sy.latex(fully_expr)
        except:
            fully_latex = recipe_latex

        return {
            'symbolic': self._evaluate_expression(expr, mode='symbolic'),
            'recipe_evaluated': recipe_evaluated,
            'fully_evaluated': self._evaluate_expression(expr, mode='full'),
            'missing_coefficients': missing_coefficients,
            'symbolic_latex': symbolic_latex,
            'recipe_latex': recipe_latex,
            'fully_latex': fully_latex
        }

    def run(self, host='127.0.0.1', port=5000, debug=True):
        """Run the Flask development server."""
        print(f"\n{'='*60}")
        print(f"Starting FlowProg Model Visualization Server")
        print(f"{'='*60}")
        print(f"\nOpen your browser to: http://{host}:{port}")
        print(f"\nProcesses: {len(self.model.processes)}")
        print(f"Objects: {len(self.model.objects)}")
        print(f"\nPress Ctrl+C to stop the server\n")

        self.app.run(host=host, port=port, debug=debug)


def run_visualization_server(model, recipe_data=None, parameter_values=None,
                            host='127.0.0.1', port=5000, debug=True):
    """
    Run the visualization server for a flowprog model.

    Args:
        model: flowprog Model instance
        recipe_data: Dictionary of recipe coefficients
        parameter_values: Dictionary of parameter values
        host: Server host (default: '127.0.0.1')
        port: Server port (default: 5000)
        debug: Enable debug mode (default: True)
    """
    server = VisualizationServer(model, recipe_data, parameter_values)
    server.run(host=host, port=port, debug=debug)
