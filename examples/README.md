# FlowProg Examples

## Interactive Visualization Demo

### visualize_demo.py

A demonstration of the interactive model visualization tool. This creates a simple energy model and launches a web-based visualization server.

**Run the demo:**

```bash
python examples/visualize_demo.py
```

Then open your browser to http://127.0.0.1:5000

**The demo model includes:**
- 6 processes: Wind turbines, gas power plants, steel production (2 alternatives), hydrogen production, electric vehicles
- 5 objects: Electricity, natural gas, hydrogen, steel, transport services
- Allocation between alternative steel production processes
- Merit-order dispatch (wind first, gas second)
- Symbolic parameters for demand, allocation factors, and capacities

**Interactive features:**
- Click on process nodes (blue rectangles) to see X and Y expressions with full history
- Click on flow edges (arrows) to see flow expressions
- All expressions show intermediate variable breakdowns
- History labels show which modeling steps contributed to each value

See [docs/visualization.md](../docs/visualization.md) for full documentation.
