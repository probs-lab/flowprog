from .model_structure import Process, Object, ElementaryExchange, ModelStructure
from .model_builder import ModelBuilder
from .activities import AdditionalActivity, merge_activities
from .backends.sympy import SympyModel
from .boundary_processes import (
    BoundaryProcess,
    Import,
    Export,
    Source,
    Sink,
    add_boundary_processes,
)

_base_all = [
    "Process",
    "Object",
    "ElementaryExchange",
    "ModelStructure",
    "ModelBuilder",
    "AdditionalActivity",
    "merge_activities",
    "SympyModel",
    "BoundaryProcess",
    "Import",
    "Export",
    "Source",
    "Sink",
    "add_boundary_processes",
]

try:
    from .backends.numpyro import NumpyroModel
    __all__ = _base_all + ["NumpyroModel"]
except ImportError:
    __all__ = _base_all
