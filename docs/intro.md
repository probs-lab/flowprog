# flowprog

This project helps with building Material Flow (MFA) models by defining equations for system variables (process recipes, stocks, flows etc) as [Sympy](https://www.sympy.org/) expressions, which can then be evaluated later to produce a consistent set of flows for e.g. different time periods, regions, or parameter values.

```{tableofcontents}
```

## The Problem: Non-linear Systems in a Linear World

Traditional Material Flow Analysis (MFA) and Life Cycle Assessment (LCA) tools rely on **linear mathematics**: fixed process coefficients, constant market shares, and matrix-based calculations. While this works well for marginal changes, real-world systems are **fundamentally non-linear**. They exhibit:

- **Discontinuities**: Switching between production pathways when capacity limits are reached
- **Capacity constraints**: Production facilities can only produce so much
- **Conditional logic**: "Use recycled material first, then virgin production if needed"
- **Merit-order dispatch**: Prioritizing cheaper or cleaner options before fallback alternatives
- **Systemic transformations**: Circular economy transitions that change the structure of material flows

These non-linear behaviors are critical for understanding real-world scenarios like:
- Decarbonization pathways where bio-based feedstocks have limited capacity
- Circular economy transitions where recycling rates affect virgin material demand
- Industrial systems where co-products from one process supply another

Traditional linear models either **cannot represent these behaviors** or require awkward workarounds that hide assumptions and make results difficult to interpret.

## The Solution: Flow Programming

**Flow programming** is a modeling paradigm that treats system specification as a programming task. Instead of solving simultaneous equations, you **build up the model step-by-step** through a series of operations:

1. **Pull production** backwards through supply chains from demand
2. **Push consumption** forwards from available supplies
3. **Apply capacity limits** to constrain production
4. **Check balances** and fill gaps with additional production
5. **Allocate across alternatives** using market shares that can vary with conditions

Each step adds to the **model state** incrementally, creating symbolic equations that capture the full logic of your system. The result is a set of equations that can be:
- **Evaluated** with different parameters for scenarios
- **Sampled** for uncertainty analysis while preserving mass balance
- **Inspected** to understand how values were calculated
- **Integrated** with Python ecosystem tools (SALib, floweaver, Brightway)

This approach makes **non-linear, discontinuous, and conditional behavior** explicit and tractable, bridging the gap between simple matrix models and complex optimization frameworks.

## Getting started

Install `flowprog` -- it is not yet released on PyPI so this needs to be done locally or via git.

See the [examples]() to get an idea of how flowprog can be used.

See the [API documentation]() for details of the functions available.

## Why use flowprog?

Compared to alternative ways of forumulating MFA models such as using an Input-Output style model (e.g. based on solving linear systems of process input/output equations along the lines of $\boldsymbol{X} = \boldsymbol{A}^{-1} \boldsymbol{y}$), this approach is much more flexible, e.g.:

- Parts of the model can be demand-driven, while other parts can be supply-driven
- Alternative supply options can be subject to capacity limits, and dispatched in a "merit order" of options
- Model logic can include conditional behavior (e.g., "if recycled supply < demand, use virgin production")
- The response to parameter changes can be non-linear or discontinuous
- Mass balance is preserved across Monte Carlo uncertainty sampling (unlike traditional LCA)

**Use flowprog when:**
- You need to model capacity constraints or merit-order dispatch
- Your system has conditional logic or switching behavior
- You're analyzing transformative changes (not just marginal adjustments)
- You need uncertainty analysis that preserves physical constraints

**Don't use flowprog if:**
- Simple linear models are sufficient for your analysis
- You're only studying marginal changes to existing systems
- You need global optimization across many decision variables

flowprog can also make use of information about MFA system structure (processes, material types, and their connections) defined using the [PRObs system structure ontology](). By reusing this structural information, it can be possible to avoid duplication and re-use work already put into documenting a system definition.
