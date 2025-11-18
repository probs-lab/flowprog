# FlowProg Interactive Visualization

An interactive web-based visualization tool for exploring flowprog models. This tool makes model expressions **explainable** by showing the graph structure of processes and objects, along with detailed breakdowns of how expressions are built up through modeling steps.

## Features

### Interactive Graph Visualization

- **Process nodes** (blue rectangles): Represent transformation processes
- **Object nodes** (circles): Represent materials/energy/services flowing through the system
  - Red circles: Market objects (external demand/supply)
  - Gray circles: Intermediate objects (internal flows)
- **Flow edges** (arrows): Represent material/energy flows between processes and objects

### Click to Explore

#### Process Details
Click on any process to see:
- **Input and output flows** with their evaluated values
- **X (Process Input Magnitude)** expression:
  - Final symbolic expression
  - LaTeX rendering for readability
  - History of modeling steps that contributed to this value
  - Breakdown of intermediate variables with descriptions
- **Y (Process Output Magnitude)** expression:
  - Same detailed breakdown as X

#### Flow Details
Click on any flow edge to see:
- Flow type (production or consumption)
- Source and target nodes
- **Complete expression** for the flow value
- **Intermediate variable breakdown** showing how the expression was built
- Evaluated numeric value (if parameters are provided)

### Expression Breakdown

The tool shows how complex expressions are built up from simpler components:

1. **Final Expression**: The complete symbolic expression
2. **LaTeX Rendering**: Mathematical notation for clarity
3. **Modeling History**: Labels from each `model.add(..., label="...")` call that affected this expression
4. **Intermediate Variables**: Each intermediate symbol (`_x0`, `_x1`, etc.) with:
   - Symbol name
   - Description of what it represents
   - Full expression it evaluates to
   - LaTeX rendering

This makes it easy to understand:
- Why a process has a certain activity level
- How demand propagates backwards through the system
- How supply fills deficits
- Where allocation factors are applied
- How limits and constraints affect the model

## Installation

Install the required dependencies:

```bash
pip install flask flask-cors
```

Or add to your `pyproject.toml` dev dependencies:

```toml
[tool.poetry.group.dev.dependencies]
flask = "^3.0.0"
flask-cors = "^4.0.0"
```

## Usage

### Basic Usage

```python
from flowprog.imperative_model import Model, Process, Object
from flowprog.visualization import run_visualization_server

# Create your model
model = Model(processes, objects)
# ... build model with model.add() calls ...

# Define recipe data and parameters
recipe_data = {
    model.S[0, 0]: 1.0,
    model.U[1, 0]: 0.5,
    # ...
}

parameter_values = {
    demand_symbol: 100.0,
    # ...
}

# Start the visualization server
run_visualization_server(
    model=model,
    recipe_data=recipe_data,
    parameter_values=parameter_values,
    host='127.0.0.1',
    port=5000
)
```

Then open your browser to `http://127.0.0.1:5000`

### Demo Script

A complete demo is available in `examples/visualize_demo.py`:

```bash
python examples/visualize_demo.py
```

This creates a simple energy model with:
- Steel production (two alternative processes)
- Electric vehicle transport
- Electricity supply from wind and gas
- Hydrogen production
- Market-based allocation

## Architecture

The visualization tool is designed for easy migration to JupyterLab:

### Current Implementation (Standalone)
- **Backend**: Flask web server with REST API
- **Frontend**: Cytoscape.js for graph visualization
- **Math Rendering**: MathJax for LaTeX expressions

### Future JupyterLab Integration
The architecture supports migration to JupyterLab widgets:
- **ipycytoscape**: Jupyter widget version of Cytoscape.js
- Core analysis logic in `ExpressionAnalyzer` class is widget-agnostic
- API endpoints can be adapted to widget message passing

### File Structure

```
src/flowprog/visualization/
├── __init__.py              # Public API
├── server.py                # Flask server and API endpoints
├── expression_analyzer.py   # Expression decomposition logic
├── templates/
│   └── index.html          # Web UI template
└── static/
    └── app.js              # Frontend JavaScript
```

## API Endpoints

The Flask server provides the following REST API:

- `GET /` - Main visualization page
- `GET /api/graph` - Complete graph structure (nodes and edges)
- `GET /api/process/<process_id>` - Detailed process information
- `GET /api/flow/<source>/<target>/<material>` - Flow details
- `GET /api/parameters` - Current parameter values
- `POST /api/parameters` - Update parameter values

## Development

### Adding New Features

The modular architecture makes it easy to extend:

1. **New analysis features**: Add methods to `ExpressionAnalyzer`
2. **New API endpoints**: Add routes in `VisualizationServer`
3. **New UI components**: Modify `templates/index.html` and `static/app.js`

### Testing

The demo script serves as an integration test. Run it to verify all components work together:

```bash
python examples/visualize_demo.py
```

You should see:
1. Model creation output
2. Server startup messages
3. Server running on http://127.0.0.1:5000

Open the URL in your browser and test:
- Graph rendering
- Process click interactions
- Flow click interactions
- Expression display with LaTeX
- History labels
- Intermediate variable breakdown

## Troubleshooting

### Server won't start
- Check that Flask and flask-cors are installed
- Ensure port 5000 is not already in use
- Try a different port: `run_visualization_server(..., port=8000)`

### Graph not rendering
- Check browser console for JavaScript errors
- Ensure you have internet connection (CDN dependencies)
- Try a different browser

### Expressions not showing
- Verify that recipe_data contains all necessary coefficients
- Check that model.add() calls include label parameters
- Ensure intermediate symbols are being created

### Math rendering issues
- MathJax loads from CDN, requires internet connection
- Check browser console for MathJax errors
- Some complex SymPy expressions may need special LaTeX formatting

## Future Enhancements

Planned improvements:

1. **JupyterLab widget**: Native Jupyter integration with ipycytoscape
2. **Parameter sliders**: Interactive UI to adjust parameters and see real-time updates
3. **Expression tree view**: Hierarchical visualization of expression structure
4. **Export capabilities**: Save graphs as images or expressions as LaTeX
5. **Comparison mode**: Compare multiple scenarios side-by-side
6. **Performance optimization**: Handle larger models efficiently
7. **Custom styling**: User-configurable colors and layouts
8. **Animation**: Visualize flow propagation through the model

## Contributing

The visualization tool is designed to make flowprog models more understandable and debuggable. Contributions welcome for:

- UI/UX improvements
- Additional analysis features
- Performance optimizations
- JupyterLab widget implementation
- Documentation and examples

## License

Same as flowprog (MIT License)
