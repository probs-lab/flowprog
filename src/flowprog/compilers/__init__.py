"""Compilers for flowprog model steps.

Each compiler takes a ModelStructure and a list of AdditionalActivity steps,
and produces a backend-specific output.

Available compilers:

- ``sympy``: Compiles steps into accumulated SymPy expressions (the default).
- ``markdown``: Compiles steps into a human-readable Markdown description.
- ``numpyro``: Compiles steps into a NumPyro model function for Bayesian inference.
"""

from .sympy import compile_sympy
from .markdown import compile_markdown
from .numpyro import compile_numpyro, Observation, ModelSpec

__all__ = [
    "compile_sympy",
    "compile_markdown",
    "compile_numpyro",
    "Observation",
    "ModelSpec",
]
