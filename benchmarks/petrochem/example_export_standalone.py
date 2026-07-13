#!/usr/bin/env python3
"""One-off export of model structure + recipe + benchmark scenarios to plain
JSON, for building a self-contained (no-RDF) version of this model as a
flowprog test case. Not part of the regular pipeline -- run manually and copy
the result wherever the standalone version lives.
"""

import json
from pathlib import Path

import sympy as sy

from load_levers import read_levers
import load_model
from regression_snapshot import TIME_INDEX, build_scenarios, SNAPSHOT_PATH

OUT_PATH = Path(__file__).parent / "standalone_export.json"


def export_structure(model_builder, recipe_data):
    structure = model_builder.structure

    processes = [
        {"id": p.id, "produces": p.produces, "consumes": p.consumes}
        for p in structure.processes
    ]
    objects = [
        {"id": o.id, "has_market": o.has_market} for o in structure.objects
    ]

    recipe = []
    for sym, value in recipe_data.items():
        base = sym.base
        i, j = sym.indices
        role = "produces" if base == structure.S else "consumes"
        assert base == structure.S or base == structure.U
        recipe.append(
            {
                "process": structure.processes[j].id,
                "object": structure.objects[i].id,
                "role": role,
                "value": value,
            }
        )

    return {"processes": processes, "objects": objects, "recipe": recipe}


def export_scenarios(levers):
    with open(SNAPSHOT_PATH) as f:
        golden = json.load(f)
    assert golden["time_index"] == TIME_INDEX

    scenarios = build_scenarios(levers)
    assert set(scenarios) == set(golden["scenarios"])

    out = {}
    for name, settings in scenarios.items():
        params = levers.get_params(settings, time_index=TIME_INDEX)
        out[name] = {
            "params": params,
            "results": golden["scenarios"][name]["results"],
        }
    return out


def export_process_groups(levers):
    """Process id lists derived from the lever parameter declarations.
    `processes_with_elec_req` is needed by structure.build_structure() (to
    add Electricity/ProcessHeat consumption to the right processes);
    `processes_with_process_emissions` is needed by
    model_polymers.define_polymer_model()'s `processes_with_direct_emissions`
    arg.
    """
    processes_with_elec_req = [
        param.symbol[len("ElecReq_") :]
        for param in levers.params
        if param.symbol.startswith("ElecReq_")
    ]
    processes_with_process_emissions = sorted(
        {
            param.symbol[len("DirProcEmis_CH4_") :]
            for param in levers.params
            if param.symbol.startswith("DirProcEmis_")
        }
    )
    return processes_with_elec_req, processes_with_process_emissions


def main():
    levers = read_levers("levers.xlsx")
    model_data = load_model.load_model()
    model_builder, recipe_data = load_model.build_model(model_data)

    processes_with_elec_req, processes_with_process_emissions = export_process_groups(
        levers
    )

    data = {
        "structure": export_structure(model_builder, recipe_data),
        "scenarios": export_scenarios(levers),
        "processes_with_elec_req": processes_with_elec_req,
        "processes_with_process_emissions": processes_with_process_emissions,
    }

    with open(OUT_PATH, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)

    print(f"Wrote {OUT_PATH}")
    print(f"  processes: {len(data['structure']['processes'])}")
    print(f"  objects:   {len(data['structure']['objects'])}")
    print(f"  recipe entries: {len(data['structure']['recipe'])}")
    print(f"  scenarios: {len(data['scenarios'])}")


if __name__ == "__main__":
    main()
