# flowprog

This project helps with building Material Flow (MFA) models by defining equations for system variables (process recipes, stocks, flows etc) as [Sympy](https://www.sympy.org/) expressions, which can then be evaluated later to produce a consistent set of flows for e.g. different time periods, regions, or parameter values.

```{tableofcontents}
```

## Getting started

Install `flowprog` -- it is not yet released on PyPI so this needs to be done locally or via git.

See the [examples]() to get an idea of how flowprog can be used.

See the [API documentation]() for details of the functions available.

## Why use flowprog?

Compared to alternative ways of forumulating MFA models such as using an Input-Output style model (e.g. based on solving linear systems of process input/output equations along the lines of $\boldsymbol{X} = \boldsymbol{A}^{-1} \boldsymbol{y}$), this approach is much more flexible, e.g.:

- Parts of the model can be demand-driven, while other parts can be supply-driven
- Alternative supply options can be subject to capacity limits, and dispatched in a "merit order" of options

If this flexibility is not required, you may not want to use flowprog.

flowprog can also make use of information about MFA system structure (processes, material types, and their connections) defined using the [PRObs system structure ontology](). By reusing this structural information, it can be possible to avoid duplication and re-use work already put into documenting a system definition.
