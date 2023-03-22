import pandas as pd
from flowprog.imperative_model import *
from flowprog.load_from_rdf import *

from rdflib import Namespace, Graph

rdf_data_path = "system-definitions/_build/probs_rdf/output.ttl"
model_def_path = "model.ttl"

MODEL_NS = Namespace("http://c-thru.org/analyses/calculator/model/")
model_uri = MODEL_NS["Model"]

g = Graph()
g.parse(rdf_data_path, format="ttl")
g.parse(model_def_path, format="ttl")
model, recipe_data = query_model_from_endpoint(g, model_uri)

demand_symbols = {
    "Steel": sy.Symbol("Z_1"),
    "Transport": sy.Symbol("Z_2"),
}
steel_fraction = sy.Symbol("a_1")

model.add(model.pull_production(
    "Steel", demand_symbols["Steel"], until_objects=["Electricity"],
    allocate_backwards={
        "Steel": {
            "SteelProductionEAF": steel_fraction,
            "SteelProductionH2DRI": 1 - steel_fraction,
        },
    }
), label="Demand for steel")

model.add(model.pull_production(
    "TransportService", demand_symbols["Transport"], until_objects=["Electricity"],
), label="Demand for transport")


# Add wind turbine supply, up to limit of demand
wt_supply = model.pull_process_output(
    "WindTurbine",
    "Electricity",
    sy.Symbol("S_1"),
)
wt_supply_limited = model.limit(
    wt_supply,
    expr=model.expr("ProcessOutput", process="WindTurbine", object="Electricity"),
    limit=model.expr("Consumption", object="Electricity"),
)
model.add(wt_supply_limited, label="Supply from wind turbines (first choice)")

model.add(
    model.pull_process_output(
        "CCGT",
        "Electricity",
        model.object_production_deficit("Electricity")
    ),
    label="Supply from CCGT (second choice)"
)

# model.fill_blanks(fill_value=0)

# def subs_dict(d, values):
#     return {k: v.subs(values) for k, v in d.items()}

def solution_to_flows(m, values):
    all_values = {**recipe_data, **values}
    return m.to_flows(all_values)


flows_sym = model.to_flows(recipe_data)

# post = {
#     "NG demand": flows_sym.query("source == 'NaturalGas'")["value"].sum(),
#     "CO2": flows_sym.query("target == 'AtmosphericCO2'")["value"].sum(),
# }

# emissions = {
#     "Urea": flows_sym.
# }
