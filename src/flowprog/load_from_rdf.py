"""Compatibility alias for :mod:`flowprog.io.rdf`.

This module moved to :mod:`flowprog.io.rdf` when the loaders were gathered into the
:mod:`flowprog.io` subpackage, and `query_model_from_endpoint` was renamed to
`load_structure`. Both old names keep working; new code should use::

    from flowprog.io.rdf import load_structure
"""

from .io.rdf import query_model_from_endpoint

__all__ = ["query_model_from_endpoint"]
