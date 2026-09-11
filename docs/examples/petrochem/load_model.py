"""Build the petrochemicals model at the baseline scenario's parameters.

The model structure, recipe data and scenario parameters live in
`benchmarks/petrochem/`.
"""

import sys
from pathlib import Path


def _find_benchmark_model():
    for base in [Path.cwd(), *Path.cwd().parents]:
        candidate = base / "benchmarks" / "petrochem"
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "cannot find benchmarks/petrochem: run this from within the flowprog "
        "repository"
    )


sys.path.insert(0, str(_find_benchmark_model()))

from structure import load_data, build_structure  # noqa: E402
from model import define_model  # noqa: E402
from model_polymers import GWP_ALL as GWP  # noqa: E402

data = load_data()
builder, recipe_data = build_structure(data)
define_model(builder, recipe_data, data["processes_with_process_emissions"])

model = builder.build(recipe_data)
structure = model.structure
params = data["scenarios"]["baseline"]["params"]
