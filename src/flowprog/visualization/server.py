"""
Flask server for FlowProg visualization.

This module provides a web server for visualizing process flow models
with interactive exploration of expressions and evaluation modes.
"""

from flask import Flask, render_template, jsonify
from flask_cors import CORS
from typing import Dict, Optional
import traceback


class VisualizationServer:
    """
    Flask server for visualizing process flow models.

    Args:
        model: The process flow model to visualize
        recipe_data: Optional recipe coefficients for evaluation
        parameter_values: Optional parameter values for evaluation
        dev_mode: If True, use Vite dev server for frontend; if False, serve built assets
    """

    def __init__(self, model, recipe_data: Optional[Dict] = None,
                 parameter_values: Optional[Dict] = None, dev_mode: bool = False):
        self.model = model
        self.recipe_data = recipe_data or {}
        self.parameter_values = parameter_values or {}

        self.app = Flask(__name__,
                        template_folder='templates',
                        static_folder='static')
        self.app.config['ENV'] = 'development' if dev_mode else 'production'
        CORS(self.app)

        self._register_routes()

    def _register_routes(self):
        """Register Flask routes."""

        @self.app.route('/')
        def index():
            """Serve the main visualization page."""
            return render_template('index.html')

        @self.app.route('/api/graph')
        def get_graph():
            """Get the complete graph structure (nodes and edges)."""
            try:
                nodes = []
                edges = []

                # Add process nodes
                for proc in self.model.processes:
                    nodes.append({
                        'data': {
                            'id': proc.name,
                            'label': proc.name,
                            'type': 'process'
                        },
                        'classes': 'process'
                    })

                # Add object nodes
                for obj in self.model.objects:
                    obj_class = 'market' if hasattr(obj, 'is_market') and obj.is_market else 'object'
                    nodes.append({
                        'data': {
                            'id': obj.name,
                            'label': obj.name,
                            'type': 'object'
                        },
                        'classes': obj_class
                    })

                # Add edges for flows
                for flow in self.model.flows:
                    # Get source and target
                    if hasattr(flow, 'from_process') and flow.from_process:
                        source = flow.from_process.name
                    elif hasattr(flow, 'from_object') and flow.from_object:
                        source = flow.from_object.name
                    else:
                        continue

                    if hasattr(flow, 'to_process') and flow.to_process:
                        target = flow.to_process.name
                    elif hasattr(flow, 'to_object') and flow.to_object:
                        target = flow.to_object.name
                    else:
                        continue

                    # Get material name
                    material = flow.object_type.name if hasattr(flow, 'object_type') else 'unknown'

                    # Create edge ID
                    edge_id = f"{source}-{target}-{material}"

                    edges.append({
                        'data': {
                            'id': edge_id,
                            'source': source,
                            'target': target,
                            'material': material,
                            'label': material
                        },
                        'classes': 'flow'
                    })

                return jsonify({
                    'nodes': nodes,
                    'edges': edges
                })

            except Exception as e:
                traceback.print_exc()
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/process/<process_id>')
        def get_process(process_id):
            """Get detailed information about a process."""
            try:
                # Find the process
                process = next((p for p in self.model.processes if p.name == process_id), None)
                if not process:
                    return jsonify({'error': f'Process {process_id} not found'}), 404

                # Get process information
                has_stock = hasattr(process, 'has_stock') and process.has_stock

                # Get consumed and produced objects
                consumes = []
                produces = []

                if hasattr(process, 'inputs'):
                    consumes = [inp.object_type.name for inp in process.inputs]
                if hasattr(process, 'outputs'):
                    produces = [out.object_type.name for out in process.outputs]

                # Get input and output flows
                inputs = []
                outputs = []

                for flow in self.model.flows:
                    if hasattr(flow, 'to_process') and flow.to_process == process:
                        obj_name = flow.object_type.name if hasattr(flow, 'object_type') else 'unknown'
                        inputs.append({
                            'object': obj_name,
                            'value': str(flow.amount) if hasattr(flow, 'amount') else 'N/A'
                        })
                    elif hasattr(flow, 'from_process') and flow.from_process == process:
                        obj_name = flow.object_type.name if hasattr(flow, 'object_type') else 'unknown'
                        outputs.append({
                            'object': obj_name,
                            'value': str(flow.amount) if hasattr(flow, 'amount') else 'N/A'
                        })

                # Get X and Y analysis (placeholder - adapt based on actual model structure)
                x_analysis = self._get_expression_analysis(process, 'x')
                y_analysis = self._get_expression_analysis(process, 'y')

                return jsonify({
                    'process_id': process_id,
                    'has_stock': has_stock,
                    'consumes': consumes,
                    'produces': produces,
                    'inputs': inputs,
                    'outputs': outputs,
                    'x_analysis': x_analysis,
                    'y_analysis': y_analysis
                })

            except Exception as e:
                traceback.print_exc()
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/flow/<source>/<target>/<material>')
        def get_flow(source, target, material):
            """Get detailed information about a flow."""
            try:
                # Find the flow
                flow = None
                for f in self.model.flows:
                    f_source = None
                    f_target = None
                    f_material = None

                    if hasattr(f, 'from_process') and f.from_process:
                        f_source = f.from_process.name
                    elif hasattr(f, 'from_object') and f.from_object:
                        f_source = f.from_object.name

                    if hasattr(f, 'to_process') and f.to_process:
                        f_target = f.to_process.name
                    elif hasattr(f, 'to_object') and f.to_object:
                        f_target = f.to_object.name

                    if hasattr(f, 'object_type'):
                        f_material = f.object_type.name

                    if f_source == source and f_target == target and f_material == material:
                        flow = f
                        break

                if not flow:
                    return jsonify({'error': f'Flow {source} -> {target} ({material}) not found'}), 404

                # Determine flow type
                flow_type = 'unknown'
                if hasattr(flow, 'from_process') and hasattr(flow, 'to_process'):
                    flow_type = 'process-to-process'
                elif hasattr(flow, 'from_object') and hasattr(flow, 'to_process'):
                    flow_type = 'input'
                elif hasattr(flow, 'from_process') and hasattr(flow, 'to_object'):
                    flow_type = 'output'

                # Get evaluated value
                evaluated_value = str(flow.amount) if hasattr(flow, 'amount') else 'N/A'

                # Get expression analysis
                analysis = self._get_flow_expression_analysis(flow)

                return jsonify({
                    'material': material,
                    'flow_type': flow_type,
                    'source': source,
                    'target': target,
                    'evaluated_value': evaluated_value,
                    'description': f'Flow of {material} from {source} to {target}',
                    'analysis': analysis
                })

            except Exception as e:
                traceback.print_exc()
                return jsonify({'error': str(e)}), 500

    def _get_expression_analysis(self, process, variable: str) -> Dict:
        """
        Get expression analysis for a process variable (X or Y).

        This is a placeholder implementation. Adapt based on your actual
        expression analysis implementation.
        """
        # Placeholder - return basic structure
        return {
            'final_expression': f'{variable}_{process.name}',
            'final_latex': f'{variable}_{{{process.name}}}',
            'evaluation_modes': {
                'symbolic': f'{variable}_{process.name}',
                'symbolic_latex': f'{variable}_{{{process.name}}}',
                'recipe_evaluated': f'{variable}_{process.name}',
                'recipe_latex': f'{variable}_{{{process.name}}}',
                'fully_evaluated': '0',
                'fully_latex': '0',
                'missing_coefficients': []
            },
            'history': [
                f'Created variable {variable} for process {process.name}'
            ],
            'intermediates': []
        }

    def _get_flow_expression_analysis(self, flow) -> Dict:
        """
        Get expression analysis for a flow.

        This is a placeholder implementation. Adapt based on your actual
        expression analysis implementation.
        """
        material = flow.object_type.name if hasattr(flow, 'object_type') else 'unknown'

        return {
            'final_expression': f'flow_{material}',
            'final_latex': f'f_{{{material}}}',
            'evaluation_modes': {
                'symbolic': f'flow_{material}',
                'symbolic_latex': f'f_{{{material}}}',
                'recipe_evaluated': f'flow_{material}',
                'recipe_latex': f'f_{{{material}}}',
                'fully_evaluated': str(flow.amount) if hasattr(flow, 'amount') else '0',
                'fully_latex': str(flow.amount) if hasattr(flow, 'amount') else '0',
                'missing_coefficients': []
            },
            'history': [
                f'Created flow for {material}'
            ],
            'intermediates': []
        }

    def run(self, host='127.0.0.1', port=5000, debug=True, dev_mode=None):
        """
        Run the Flask development server.

        Args:
            host: Server host
            port: Server port
            debug: Enable Flask debug mode
            dev_mode: If True, serve Vue dev server; if False, serve built assets.
                     If None, use the value from __init__.
        """
        if dev_mode is not None:
            self.app.config['ENV'] = 'development' if dev_mode else 'production'

        is_dev = self.app.config['ENV'] == 'development'

        print(f"\n{'='*60}")
        print(f"Starting FlowProg Model Visualization Server")
        print(f"{'='*60}")

        if is_dev:
            print(f"\n🔧 Development Mode")
            print(f"Vue Frontend: http://localhost:3000")
            print(f"Flask Backend: http://{host}:{port}")
            print(f"\nMake sure to run 'npm run dev' in the frontend directory!")
        else:
            print(f"\n📦 Production Mode")
            print(f"Open your browser to: http://{host}:{port}")

        print(f"\nProcesses: {len(self.model.processes)}")
        print(f"Objects: {len(self.model.objects)}")
        print(f"\nPress Ctrl+C to stop the server\n")

        self.app.run(host=host, port=port, debug=debug)
