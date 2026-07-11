"""Load the petrochemicals model structure, recipe, and benchmark scenarios
from model_data.json.

model_data.json was extracted once from the C-THRU Global Petrochemicals
Calculator (which normally builds this structure from RDF via
flowprog.load_from_rdf) -- see README.md.
"""

import json
from pathlib import Path

from rdflib import URIRef

from flowprog import Process, Object, ModelBuilder

DATA_PATH = Path(__file__).parent / "model_data.json"
MASS = URIRef("http://qudt.org/vocab/quantitykind/Mass")


def load_data() -> dict:
    with open(DATA_PATH) as f:
        return json.load(f)


def build_structure(data: dict) -> tuple[ModelBuilder, dict]:
    """Return (ModelBuilder, recipe_data) for the model in `data`."""
    struct = data["structure"]

    processes = [
        Process(id=p["id"], produces=p["produces"], consumes=p["consumes"])
        for p in struct["processes"]
    ]
    objects = [
        Object(id=o["id"], metric=MASS, has_market=o["has_market"])
        for o in struct["objects"]
    ]

    builder = ModelBuilder(processes, objects)

    recipe_data = {}
    for entry in struct["recipe"]:
        i = builder.structure.lookup_object(entry["object"])
        j = builder.structure.lookup_process(entry["process"])
        base = builder.S if entry["role"] == "produces" else builder.U
        recipe_data[base[i, j]] = entry["value"]

    return builder, recipe_data
