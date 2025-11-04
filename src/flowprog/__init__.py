"""
Flowprog - Define material flows models as programs

Main modules:
- imperative_model: Core Model, Process, and Object classes
- load_from_rdf: Load models from RDF data
- milp_transform: Transform models to MILP for optimization (requires mip package)
- milp_solvers: Solver backends for MILP problems (requires mip package)
"""

from flowprog.imperative_model import Model, Process, Object

__all__ = [
    "Model",
    "Process",
    "Object",
]

# Conditionally import MILP modules if available
try:
    from flowprog.milp_transform import (
        MILPTransformer,
        MILPModel,
        BoundsAnalyzer,
        PiecewiseLinearizer,
    )
    from flowprog.milp_solvers import (
        PythonMIPBackend,
        DictExportBackend,
        get_available_solvers,
        get_recommended_backend,
        SolverConfig,
        SolverSolution,
    )

    __all__.extend([
        "MILPTransformer",
        "MILPModel",
        "BoundsAnalyzer",
        "PiecewiseLinearizer",
        "PythonMIPBackend",
        "DictExportBackend",
        "get_available_solvers",
        "get_recommended_backend",
        "SolverConfig",
        "SolverSolution",
    ])

    _MILP_AVAILABLE = True

except ImportError:
    _MILP_AVAILABLE = False


def has_milp_support() -> bool:
    """Check if MILP transformation and solving is available."""
    return _MILP_AVAILABLE


__version__ = "0.1.0"
