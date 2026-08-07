"""Loading model structures from external definitions of a system.

Each module here exposes a `load_structure` function returning
`(structure, recipe_data)` -- a :class:`~flowprog.model_structure.ModelStructure`
ready for :class:`~flowprog.model_builder.ModelBuilder`, and the recipe coefficients
to `build` it with:

- :mod:`flowprog.io.rdf` reads RDF produced by a Sphinx build of the system
  definitions (or any other source of PRObs-shaped RDF).
- :mod:`flowprog.io.system_definitions` reads the ``system:process`` /
  ``system:object`` directive sources directly, with no Sphinx build in between.

The modules are not imported here, so that importing `flowprog.io` does not require
the optional dependencies of either.
"""

__all__: list[str] = []
