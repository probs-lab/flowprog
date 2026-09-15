# flowprog: procedural generation of material flow model

This Python package defines a framework for building up mass flow models using Sympy symbolic equations.

## Getting started

Install `flowprog` from PyPI. For example, using [uv](https://docs.astral.sh/uv):

``` shellsession
uv add flowprog
```

or using pip (with a suitable virtual environment activated):

``` shellsession
pip install flowprog
```

See the documentation in `docs/` for more details and examples.

## Developing

If you don't have [uv installed already, install it](https://docs.astral.sh/uv/getting-started/installation/).

Then install dependencies using `uv`:

``` shellsession
uv sync
```

Build the documentation and examples using Jupyter Book:

``` shellsession
uv run jb build docs
```

Then open the resulting HTML files in `docs/_build/html/index.html` in your browser.

Run the tests using `pytest`:

``` shellsession
uv run pytest tests
```
