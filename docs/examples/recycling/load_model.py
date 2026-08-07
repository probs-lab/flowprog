import pandas as pd
import sympy as sy
from flowprog import ModelBuilder
from flowprog.io.system_definitions import load_structure

# The processes and objects are read straight from the definitions, so editing
# system-definitions.md and re-running this file is enough to see the change.
#
# Which processes make up the model, and which objects need markets, are worked out
# from the definitions too: every process with a recipe is included, and an object
# gets a market if the model both produces and consumes it -- here EoLTechnology,
# PCBs and GoldAndPlastic. PureGold, OtherParts and MixedPCBWaste are only produced,
# so they leave the model boundary unbalanced.
model_structure, recipe_data = load_structure("system-definitions.md")
builder = ModelBuilder.from_structure(model_structure)

## Define the symbols

eol_flow = sy.Symbol("F")
process1_gold_capacity = sy.Symbol("C")

## Define the model logic


## Step 1: EoL flows defined by dMFA, up to PCBs (where the first capacity-limited logic happens)

builder.add(
    builder.push_process_input("Disassembly", "EoLTechnology", eol_flow, until_objects={"PCBs"}),
    label="Step 1 EOL flow"
)

## Step 2: Handle PCBs via Process 1 up to capacity limit

proposal = builder.push_process_input(
    "PCBProcess1", "PCBs", builder.object_consumption_deficit("PCBs")
)
limited = builder.limit(
    proposal,
    expr=builder.expr("ProcessOutput", process_id="PCBProcess1", object_id="PureGold"),
    limit=process1_gold_capacity
)
builder.add(
    limited,
    label="Step 2 PCBs via Process1"
)

# To be completed...

model = builder.build(recipe_data)
flows_sym = model.to_flows(recipe_data)
