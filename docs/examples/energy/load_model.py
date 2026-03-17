import pandas as pd
import sympy as sy
from flowprog import ModelBuilder
from flowprog.load_from_rdf import query_model_from_endpoint

from rdflib import Namespace, Graph

rdf_data_path = "../../_build/probs_rdf/output.ttl"
model_def_path = "model.ttl"

MODEL_NS = Namespace("http://probs-lab.github.io/flowprog/examples/energy-model/")
model_uri = MODEL_NS["Model"]

g = Graph()
g.parse(rdf_data_path, format="ttl")
g.parse(model_def_path, format="ttl")
model_structure, recipe_data = query_model_from_endpoint(g, model_uri)
builder = ModelBuilder.from_structure(model_structure)

demand_symbols = {
    "Steel": sy.Symbol("Z_1"),
    "Transport": sy.Symbol("Z_2"),
}
steel_fraction = sy.Symbol("a_1")

builder.add(builder.pull_production(
    "Steel", demand_symbols["Steel"], until_objects=["Electricity"],
    allocate_backwards={
        "Steel": {
            "SteelProductionEAF": steel_fraction,
            "SteelProductionH2DRI": 1 - steel_fraction,
        },
    }
), label="Demand for steel")

builder.add(builder.pull_production(
    "TransportService", demand_symbols["Transport"], until_objects=["Electricity"],
), label="Demand for transport")


# Add wind turbine supply, up to limit of demand
wt_supply = builder.pull_process_output(
    "WindTurbine",
    "Electricity",
    sy.Symbol("S_1"),
)
wt_supply_limited = builder.limit(
    wt_supply,
    expr=builder.expr("ProcessOutput", process_id="WindTurbine", object_id="Electricity"),
    limit=builder.expr("Consumption", object_id="Electricity"),
)
builder.add(wt_supply_limited, label="Supply from wind turbines (first choice)")

builder.add(
    builder.pull_process_output(
        "CCGT",
        "Electricity",
        builder.object_production_deficit("Electricity")
    ),
    label="Supply from CCGT (second choice)"
)

model = builder.build(recipe_data)
flows_sym = model.to_flows(recipe_data)


# post = {
#     "NG demand": flows_sym.query("source == 'NaturalGas'")["value"].sum(),
#     "CO2": flows_sym.query("target == 'AtmosphericCO2'")["value"].sum(),
# }

# emissions = {
#     "Urea": flows_sym.
# }
