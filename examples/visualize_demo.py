"""
Demo script for FlowProg visualization.

This script creates a simple process flow model and launches the
interactive visualization server.
"""

from flowprog.imperative_model import (
    Model, Process, Object, InputFlow, OutputFlow
)
from flowprog.visualization import VisualizationServer


def create_demo_model():
    """Create a simple demo model for visualization."""
    model = Model()

    # Create objects (materials)
    iron_ore = Object('Iron Ore')
    coal = Object('Coal')
    iron = Object('Iron')
    steel = Object('Steel')

    # Create processes
    smelt = Process('Smelt', has_stock=False)
    forge = Process('Forge', has_stock=False)

    # Add inputs and outputs to processes
    smelt.add_input(InputFlow(iron_ore, amount=2))
    smelt.add_input(InputFlow(coal, amount=1))
    smelt.add_output(OutputFlow(iron, amount=1))

    forge.add_input(InputFlow(iron, amount=3))
    forge.add_input(InputFlow(coal, amount=1))
    forge.add_output(OutputFlow(steel, amount=1))

    # Add to model
    model.add_process(smelt)
    model.add_process(forge)

    return model


if __name__ == '__main__':
    # Create the demo model
    model = create_demo_model()

    # Recipe data (optional - for expression evaluation)
    recipe_data = {
        'S_Iron_Ore_Smelt': 2,
        'S_Coal_Smelt': 1,
        'U_Iron_Smelt': 1,
        'S_Iron_Forge': 3,
        'S_Coal_Forge': 1,
        'U_Steel_Forge': 1
    }

    # Create and run the visualization server
    # Set dev_mode=True if you're running Vue dev server (npm run dev)
    # Set dev_mode=False for production (after npm run build)
    server = VisualizationServer(model, recipe_data=recipe_data, dev_mode=False)
    server.run(port=5000, debug=True)
