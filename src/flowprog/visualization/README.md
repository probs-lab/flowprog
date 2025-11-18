# FlowProg Visualization

Interactive web-based visualization of process flow models using Vue.js and Cytoscape.

## Features

- **Interactive graph visualization** using Cytoscape.js with dagre layout
- **Process and flow details panels** with expression analysis
- **Three expression evaluation modes**:
  - Fully Symbolic (no substitution)
  - Recipe Evaluated (coefficients substituted)
  - Fully Evaluated (all parameters substituted)
- **LaTeX rendering** for mathematical expressions using MathJax
- **Modeling history** and intermediate variables display
- **Responsive design** with smooth animations

## Installation

Install the required Python dependencies:

```bash
poetry install
```

The JavaScript dependencies are already included in the `frontend` directory.

## Usage

### Quick Start

```python
from flowprog.imperative_model import Model, Process, Object, InputFlow, OutputFlow
from flowprog.visualization import VisualizationServer

# Create your model
model = Model()
# ... add processes, objects, and flows ...

# Create and run the visualization server
server = VisualizationServer(model)
server.run(port=5000)
```

Then open http://localhost:5000 in your browser.

### Example

See `examples/visualize_demo.py` for a complete example:

```bash
python examples/visualize_demo.py
```

## Development

### Running in Development Mode

For active development with hot reload:

1. Start the Flask backend:
```bash
python examples/visualize_demo.py
```

2. In another terminal, start the Vue dev server:
```bash
cd src/flowprog/visualization/frontend
npm run dev
```

3. Open http://localhost:3000 (Vue dev server)

### Building for Production

Build the Vue frontend:

```bash
cd src/flowprog/visualization/frontend
npm run build
```

This outputs the built assets to `../static/dist/`.

## Architecture

The visualization consists of:

- **Backend**: Flask server (`server.py`) that serves the model data via REST API
- **Frontend**: Vue.js SPA built with Vite
  - **Components**: Modular Vue components (Graph, Panels, Expressions)
  - **Stores**: Reactive state management with Vue's Composition API
  - **Composables**: Reusable logic (API calls, MathJax rendering)

### Component Hierarchy

```
App.vue
├── GraphView.vue (Cytoscape graph)
│   └── ControlBar.vue (Fit, Reset buttons)
└── DetailsPanel.vue (Side panel)
    ├── ProcessPanel.vue (Process details)
    │   └── ExpressionModes.vue
    └── FlowPanel.vue (Flow details)
        └── ExpressionModes.vue
            ├── ExpressionBox.vue (reusable)
            ├── HistoryList.vue
            └── IntermediatesList.vue
```

## API Endpoints

- `GET /` - Serve the visualization HTML
- `GET /api/graph` - Get graph structure (nodes and edges)
- `GET /api/process/<process_id>` - Get process details
- `GET /api/flow/<source>/<target>/<material>` - Get flow details

## Customization

You can customize the visualization by modifying:

- **Graph styles**: Edit `cytoscapeStyles` in `GraphView.vue`
- **Colors and layout**: Edit component `<style>` sections
- **Expression analysis**: Override `_get_expression_analysis()` and `_get_flow_expression_analysis()` in `server.py`

## Documentation

For detailed migration plan and architecture documentation, see:
- `docs/vue-migration-plan.md` - Complete migration guide
- `docs/vue-architecture-quick-ref.md` - Quick reference

## Requirements

- Python 3.8+
- Node.js 18+
- Modern web browser with JavaScript enabled

## License

MIT
