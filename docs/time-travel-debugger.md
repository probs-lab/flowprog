# Time-Travelling Debugger

The FlowProg visualization now includes a time-travelling debugger that allows you to step through the model building process and see the state of flows and processes after each call to `model.add()`.

## Features

### 1. **Step-by-Step Navigation**
- Navigate backward and forward through model building steps
- Jump to first or last step instantly
- Use slider for quick navigation to any specific step

### 2. **Auto-Play Mode**
- Automatically advance through steps with the Play/Pause button
- Steps advance every 1.5 seconds
- Automatically stops when reaching the final step

### 3. **Flow Magnitude Visualization**
- Edge widths dynamically scale based on flow values
- Larger flows appear as thicker arrows
- Values are normalized to a 1-10 pixel range for clarity

### 4. **Step Context**
- Current step indicator shows which `model.add()` call is active
- Step labels display the description from the original `add()` call
- Example: "Step 2/4: Wind turbine supply (first choice, limited by demand)"

### 5. **State Consistency**
- All process details reflect the state at the selected step
- All flow details reflect the state at the selected step
- Graph topology shows only active flows for that step

## How It Works

### Model Snapshots
When you call `model.add(values, label="description")`, the model now automatically:
1. Captures a snapshot of the current `_values` dictionary
2. Captures a snapshot of the `_intermediates` list
3. Stores them with the step label for later retrieval

### API Endpoints
The visualization server provides time-travel endpoints:
- `/api/steps` - Get list of all available steps
- `/api/graph/<step>` - Get graph structure at specific step
- `/api/process/<step>/<process_id>` - Get process details at step
- `/api/flow/<step>/<source>/<target>/<material>` - Get flow details at step

### UI Controls
Located at the bottom center of the visualization:
- **⏮ First** - Jump to first step (step 0)
- **⏪ Prev** - Go to previous step
- **▶ Play / ⏸ Pause** - Toggle auto-play mode
- **Next ⏩** - Go to next step
- **Last ⏭** - Jump to last step
- **Slider** - Drag to navigate to any step directly

## Example Usage

```python
from flowprog.imperative_model import Model
from flowprog.visualization import run_visualization_server

# Create model
model = Model(processes, objects)

# Each add() call creates a snapshot
model.add(
    model.pull_production("Steel", steel_demand),
    label="Steel demand pulled through production"
)

model.add(
    model.pull_production("Transport", transport_demand),
    label="Transport demand pulled through EV fleet"
)

model.add(
    wind_supply,
    label="Wind turbine supply (limited by demand)"
)

# Run visualization - time-travel controls automatically appear
run_visualization_server(model, recipe_data, parameter_values)
```

## Benefits

1. **Understanding Model Building** - See how each step contributes to the final model
2. **Debugging** - Identify which step introduces unexpected behavior
3. **Education** - Demonstrate model construction process step-by-step
4. **Verification** - Confirm each step produces expected intermediate results
5. **Flow Analysis** - Compare flow magnitudes across different steps

## Technical Details

### Snapshot Storage
- Snapshots are stored in `model._snapshots` as tuples: `(label, values_dict, intermediates_list)`
- Minimal overhead: only dict/list references are stored (no deep copy until needed)
- Snapshots are captured automatically on every `add()` call

### Edge Width Calculation
```javascript
// Normalize flow values to 1-10 pixel range
const minValue = Math.min(...flowValues);
const maxValue = Math.max(...flowValues);
const range = maxValue - minValue;

const width = 1 + ((numericValue - minValue) / range) * 9;
```

### State Restoration
When viewing a historical step, the server temporarily:
1. Swaps in the snapshot's `_values` and `_intermediates`
2. Generates the graph/process/flow data
3. Restores the original values
4. Returns the historical data to the client

This ensures the model's current state is never permanently modified by time-travel operations.

## Performance Notes

- Snapshot creation adds negligible overhead to `model.add()`
- Server-side state swapping is fast (dictionary assignment)
- Frontend graph updates use Cytoscape's efficient rendering
- Large models (100+ steps) may experience slight UI lag during auto-play

## Future Enhancements

Potential improvements for future versions:
- Diff view showing what changed between steps
- Branching/forking to explore alternative model paths
- Export specific step as standalone model
- Snapshot comparison mode (side-by-side view)
- Keyboard shortcuts for step navigation
- Step bookmarking/annotations
