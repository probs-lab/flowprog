# flowprog: procedural generation of material flow model

This Python package defines a framework for building up mass flow models using Sympy symbolic equations.

## Getting started

Install `flowprog` -- it is not yet released on PyPI so this needs to be done locally or via git.

See the documentation in `docs/` for more details and examples.

## Developing

Install dependencies using `poetry`:

``` shellsession
poetry install
```

Build the documentation and examples using Jupyter Book:

``` shellsession
poetry run jb build docs
```

Then open the resulting HTML files in `docs/_build/html/index.html` in your browser.

Run the tests using `pytest`:

``` shellsession
poetry run pytest tests
```
