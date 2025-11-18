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

                edges.append({
                    'data': {
                        'id': edge_id,
                        'source': source,
                        'target': target,
                        'label': f"{material}\n{display_value}",
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
            x_expr = self.model._values.get(self.model.X[proc_idx], sy.Integer(0))
            y_expr = self.model._values.get(self.model.Y[proc_idx], sy.Integer(0))

            # Analyze expressions
            x_analysis = self.analyzer.analyze_expression(x_expr, f"X[{process_id}] (Process Input)")
            y_analysis = self.analyzer.analyze_expression(y_expr, f"Y[{process_id}] (Process Output)")

            # Get input and output flows
            inputs = []
            outputs = []

            for obj_id in process.consumes:
                obj_idx = self.model._obj_name_to_idx[obj_id]
                flow_expr = x_expr * self.model._values.get(
                    self.model.U[obj_idx, proc_idx],
                    sy.Integer(0)
                )
                inputs.append({
                    'object': obj_id,
                    'expression': str(flow_expr),
                    'latex': sy.latex(flow_expr),
                    'value': self._evaluate_expression(flow_expr)
                })

            for obj_id in process.produces:
                obj_idx = self.model._obj_name_to_idx[obj_id]
                flow_expr = y_expr * self.model._values.get(
                    self.model.S[obj_idx, proc_idx],
                    sy.Integer(0)
                )
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
                s_expr = self.model._values.get(self.model.S[obj_idx, proc_idx], sy.Integer(0))
                flow_expr = y_expr * s_expr

                flow_type = "Production"
                description = f"Production of {material} from process {process_id}"

            else:
                object_id = source
                process_id = target
                proc_idx = self.model._process_name_to_idx[process_id]
                obj_idx = self.model._obj_name_to_idx[object_id]

                # Flow = X[j] * U[i, j]
                x_expr = self.model._values.get(self.model.X[proc_idx], sy.Integer(0))
                u_expr = self.model._values.get(self.model.U[obj_idx, proc_idx], sy.Integer(0))
                flow_expr = x_expr * u_expr

                flow_type = "Consumption"
                description = f"Consumption of {material} by process {process_id}"

            # Analyze the flow expression
            analysis = self.analyzer.analyze_expression(flow_expr, description)

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

    def _evaluate_expression(self, expr) -> str:
        """Evaluate an expression with current parameter values."""
        if isinstance(expr, (int, float)):
            return f"{expr:.4g}"

        if not isinstance(expr, sy.Expr):
            return str(expr)

        # Try to evaluate with available parameters
        all_params = {**self.recipe_data, **self.parameter_values}

        try:
            # Substitute known values
            result = expr.subs(all_params)

            # If it evaluates to a number, format it
            if result.is_number:
                return f"{float(result):.4g}"
            else:
                # Return symbolic form
                return str(result)
        except:
            return str(expr)

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
