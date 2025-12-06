# Model serialisation

Model state can be saved/loaded using JSON serialisation. This allows you to:

- Share specific versions of models without requiring recipients to rerun all `model.add()` calls
- Archive models in a reproducible, text-based format
- Create checkpoints during long model-building processes
- Version control model states using git

## Basic usage

### Saving a model

```python
from flowprog.imperative_model import Model, Process, Object
from rdflib import URIRef
import sympy as sy

# Create and configure a model
MASS = URIRef("http://qudt.org/vocab/quantitykind/Mass")
processes = [Process("Production", produces=["product"], consumes=["input"])]
objects = [
    Object("input", MASS, has_market=False),
    Object("product", MASS, has_market=True),
]
model = Model(processes, objects)

# Build up the model
demand = sy.Symbol("demand", positive=True)
model.add(model.pull_production("product", demand))
model.add({model.S[1, 0]: 2.0, model.U[0, 0]: 1.0})

# Save with metadata
model.save(
    "my_model_v1.json",
    metadata={
        "description": "Production model with 2:1 conversion ratio",
        "author": "Your Name",
        "version": 1.0,
    }
)
```

### Loading a model

```python
from flowprog.imperative_model import Model

# Load the complete model state
loaded_model = Model.load("my_model_v1.json")

# The model is ready to use immediately
print(loaded_model.eval(loaded_model.Y[0]))  # Works with symbolic expressions
print(loaded_model.to_flows({demand: 100}))  # Generate flows with parameter values

# Continue building on the loaded model
loaded_model.add({loaded_model.Y[0]: 50})
```

## What gets saved

The JSON file contains:

- **Model structure**: All processes and objects with their properties
- **Symbolic expressions**: All values in `_values` dict, serialised using SymPy's `srepr()`
- **Intermediate symbols**: Symbols created during allocation and propagation
- **History**: Labels tracking how values were assigned
- **Metadata**: Custom metadata you provide (description, author, etc.)
- **Timestamp**: When the model was saved

