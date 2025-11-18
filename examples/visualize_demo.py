"""
Demo script for the interactive model visualization tool.

This script creates a simple energy model and launches the visualization server.
Click on processes to see their X and Y expressions broken down.
Click on flows to see the expression for that specific flow.
"""

import sympy as sy
from rdflib import URIRef
from flowprog.imperative_model import Model, Process, Object
from flowprog.visualization import run_visualization_server

# Define metrics
ENERGY = URIRef("http://qudt.org/vocab/quantitykind/Energy")
MASS = URIRef("http://qudt.org/vocab/quantitykind/Mass")
SERVICE = URIRef("http://example.org/Service")

# Create a simple energy model
def create_demo_model():
    """Create a demonstration energy model."""

    # Define processes
    processes = [
        Process("WindTurbine", produces=["Electricity"], consumes=[]),
        Process("CCGT", produces=["Electricity"], consumes=["NaturalGas"]),
        Process("SteelProductionEAF", produces=["Steel"], consumes=["Electricity"]),
        Process("SteelProductionH2DRI", produces=["Steel"], consumes=["Electricity", "Hydrogen"]),
        Process("HydrogenProduction", produces=["Hydrogen"], consumes=["Electricity"]),
        Process("ElectricVehicle", produces=["TransportService"], consumes=["Electricity"]),
    ]

    # Define objects
    objects = [
        Object("Electricity", ENERGY, has_market=True),
        Object("NaturalGas", ENERGY, has_market=True),
        Object("Hydrogen", ENERGY, has_market=False),
        Object("Steel", MASS, has_market=True),
        Object("TransportService", SERVICE, has_market=True),
    ]

    # Create model
    model = Model(processes, objects)

    # Define recipe data (coefficients)
    # Object indices: Electricity=0, NaturalGas=1, Hydrogen=2, Steel=3, TransportService=4
    # Process indices: WindTurbine=0, CCGT=1, SteelProductionEAF=2, SteelProductionH2DRI=3,
    #                  HydrogenProduction=4, ElectricVehicle=5
    recipe_data = {
        # Supply coefficients (S[i, j] - how much of object i is produced per unit activity of process j)
        model.S[0, 0]: 1.0,      # WindTurbine produces 1 unit Electricity
        model.S[0, 1]: 1.0,      # CCGT produces 1 unit Electricity
        model.S[3, 2]: 1.0,      # SteelProductionEAF produces 1 unit Steel (object 3)
        model.S[3, 3]: 1.0,      # SteelProductionH2DRI produces 1 unit Steel (object 3)
        model.S[2, 4]: 1.0,      # HydrogenProduction produces 1 unit Hydrogen (object 2)
        model.S[4, 5]: 1.0,      # ElectricVehicle produces 1 unit TransportService (object 4)

        # Use coefficients (U[i, j] - how much of object i is consumed per unit activity of process j)
        model.U[1, 1]: 2.5,      # CCGT consumes 2.5 units NaturalGas per unit activity
        model.U[0, 2]: 0.5,      # SteelProductionEAF consumes 0.5 units Electricity per unit Steel
        model.U[0, 3]: 0.3,      # SteelProductionH2DRI consumes 0.3 units Electricity per unit Steel
        model.U[2, 3]: 0.05,     # SteelProductionH2DRI consumes 0.05 units Hydrogen per unit Steel
        model.U[0, 4]: 50.0,     # HydrogenProduction consumes 50 units Electricity per unit Hydrogen
        model.U[0, 5]: 0.2,      # ElectricVehicle consumes 0.2 units Electricity per unit Transport
    }

    # Define demand symbols
    steel_demand = sy.Symbol("Z_steel")
    transport_demand = sy.Symbol("Z_transport")
    steel_fraction_eaf = sy.Symbol("a_eaf")  # Fraction of steel from EAF
    wind_capacity = sy.Symbol("S_wind")  # Wind turbine capacity

    # Build model by pulling demand backwards
    model.add(
        model.pull_production(
            "Steel",
            steel_demand,
            until_objects=["Electricity", "NaturalGas"],
            allocate_backwards={
                "Steel": {
                    "SteelProductionEAF": steel_fraction_eaf,
                    "SteelProductionH2DRI": 1 - steel_fraction_eaf,
                },
            }
        ),
        label="Steel demand pulled through production processes"
    )

    model.add(
        model.pull_production(
            "TransportService",
            transport_demand,
            until_objects=["Electricity", "NaturalGas"]
        ),
        label="Transport demand pulled through EV fleet"
    )

    # Add wind supply (first choice for electricity)
    wt_supply = model.pull_process_output(
        "WindTurbine",
        "Electricity",
        wind_capacity,
    )
    wt_supply_limited = model.limit(
        wt_supply,
        expr=model.expr("ProcessOutput", process_id="WindTurbine", object_id="Electricity"),
        limit=model.expr("Consumption", object_id="Electricity"),
    )
    model.add(wt_supply_limited, label="Wind turbine supply (first choice, limited by demand)")

    # Add CCGT to fill deficit (second choice for electricity)
    model.add(
        model.pull_process_output(
            "CCGT",
            "Electricity",
            model.object_production_deficit("Electricity"),
            until_objects=["NaturalGas"]
        ),
        label="CCGT supply (second choice, fills remaining electricity demand)"
    )

    # Define parameter values for evaluation
    parameter_values = {
        steel_demand: 100.0,        # 100 units of steel demand
        transport_demand: 500.0,    # 500 units of transport demand
        steel_fraction_eaf: 0.7,    # 70% of steel from EAF, 30% from H2-DRI
        wind_capacity: 80.0,        # 80 units of wind capacity
    }

    return model, recipe_data, parameter_values


if __name__ == "__main__":
    print("=" * 70)
    print("FlowProg Interactive Visualization Demo")
    print("=" * 70)
    print("\nCreating demonstration energy model...")

    model, recipe_data, parameter_values = create_demo_model()

    print(f"✓ Model created with {len(model.processes)} processes and {len(model.objects)} objects")
    print("\nProcesses:")
    for p in model.processes:
        print(f"  - {p.id}")

    print("\nObjects:")
    for o in model.objects:
        market = " (market)" if o.has_market else ""
        print(f"  - {o.id}{market}")

    print("\nParameter values:")
    for symbol, value in parameter_values.items():
        print(f"  - {symbol} = {value}")

    print("\n" + "=" * 70)
    print("Starting visualization server...")
    print("=" * 70)
    print("\nInteractive features:")
    print("  • Click on a PROCESS node (blue rectangles) to see:")
    print("    - Input and output flows")
    print("    - X (input magnitude) expression breakdown")
    print("    - Y (output magnitude) expression breakdown")
    print("    - Modeling history (which steps contributed)")
    print("    - Intermediate variable definitions")
    print()
    print("  • Click on a FLOW edge (arrows) to see:")
    print("    - Flow type (production or consumption)")
    print("    - Complete expression for the flow value")
    print("    - Breakdown of intermediate calculations")
    print()
    print("  • Red circles are market objects (external demand/supply)")
    print("  • Gray circles are intermediate objects (internal flows)")
    print("\n" + "=" * 70)

    # Run the visualization server
    run_visualization_server(
        model=model,
        recipe_data=recipe_data,
        parameter_values=parameter_values,
        host='127.0.0.1',
        port=5000,
        debug=True
    )
