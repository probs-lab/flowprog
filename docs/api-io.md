# Loading model structures

A model's structure -- its processes, objects, and recipes -- is usually written as
*system definitions*: documents using the `system:process` and `system:object`
directives, as in {doc}`examples/energy/system-definitions`.

`flowprog.io` has two ways to load them, both returning `(structure, recipe_data)`
ready to pass to {py:class}`flowprog.ModelBuilder`:

- {py:mod}`flowprog.io.system_definitions` parses the definition documents directly.
- {py:mod}`flowprog.io.rdf` reads the RDF that a Sphinx build of those same
  documents exports, via the `sphinx_probs_rdf` extension.

Loading the definitions directly is usually what you want while iterating on a model:
there is no build step in between, so editing a definition and re-running the model
is enough to see the change. Load from RDF when the model structure comes from
somewhere other than the definition documents, or is combined with other RDF data.

## From system definitions

Install the optional dependency for this, which brings in the parser:

```shellsession
pip install 'flowprog[definitions]'
```

Then:

```python
from flowprog import ModelBuilder
from flowprog.io.system_definitions import load_structure

structure, recipe_data = load_structure(["system-definitions.md"])
builder = ModelBuilder.from_structure(structure)
```

Unlike the RDF path, there is no separate model definition saying which processes
belong to the model and which objects have markets. By default both are inferred from
the definitions -- see `load_structure` below -- and either can be given explicitly:

```python
structure, recipe_data = load_structure(
    ["system-definitions.md"],
    markets=["Electricity", "Hydrogen"],
)
```

If the definitions use units beyond kg/m2/m3/-, describe them with `unit_table`. This
takes the same form as the `probs_rdf_units` setting used by the Sphinx extension:

```python
from flowprog.io.system_definitions import load_structure, unit_table

units = unit_table({
    "kWh": (1, "Energy"),
    "pkm": "<http://probs-lab.github.io/flowprog/metrics/PassengerKM>",
})
structure, recipe_data = load_structure(["system-definitions.md"], units=units)
```

A recipe side written in bare `%` is loaded when every object in that process's recipe
is measured in the same metric: the shares are then a dimensionless recipe in that
metric, divided by 100. Shares are rejected when a process mixes metrics -- relating a
side in kg to one in m3 needs a coefficient in m3/kg, which a share cannot express --
and `%basis` shares are rejected outright, since they name a basis to convert into.

```{eval-rst}
.. autofunction:: flowprog.io.system_definitions.load_structure

.. autofunction:: flowprog.io.system_definitions.unit_table

.. autofunction:: flowprog.io.system_definitions.select_processes

.. autofunction:: flowprog.io.system_definitions.select_markets

.. autofunction:: flowprog.io.system_definitions.structure_from_parsed_system
```

## From RDF

`load_structure` expects an object with a `query` method, such as an RDFlib graph or
a `rdfox_runner` `RDFoxEndpoint`, and the URI of a model within it whose
`probs:hasProcess` and `probs:hasMarketForObject` statements say what the model
contains.

```python
from rdflib import Graph, Namespace
from flowprog.io.rdf import load_structure

MODEL = Namespace("http://probs-lab.github.io/flowprog/examples/energy-model/")

g = Graph()
g.parse("_build/probs_rdf/output.ttl", format="ttl")
g.parse("model.ttl", format="ttl")
structure, recipe_data = load_structure(g, MODEL["Model"])
```

```{eval-rst}
.. autofunction:: flowprog.io.rdf.load_structure
```

```{note}
This module was `flowprog.load_from_rdf`, and `load_structure` was
`query_model_from_endpoint`. Both old names still work.
```
