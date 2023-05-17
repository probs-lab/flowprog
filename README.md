# flowprog: procedural generation of material flow model

This Python package defines a framework for building up mass flow models using Sympy symbolic equations.

## Getting started

Install `flowprog` -- it is not yet released on PyPI so this needs to be done locally or via git.

See the documentation in `docs/` for more details and examples.

## Developing

If you don't have [python-poetry installed already, install it](https://python-poetry.org/docs/#installation).
- When using Anaconda on Windows, we actually just installed poetry directly into the base anaconda environment using `pip install poetry` in an anaconda terminal.

Then install dependencies using `poetry`:

``` shellsession
poetry install
```

Build the documentation and examples using Jupyter Book:

``` shellsession
poetry run jb build docs
```

(on Windows anaconda terminal, `poetry run` didn't work on a network drive -- but first running `poetry shell` to open a new terminal with the poetry environment activated, and then just `jb build docs` should work. Or keep your files on a local drive, not a network drive)

Then open the resulting HTML files in `docs/_build/html/index.html` in your browser.

Run the tests using `pytest`:

``` shellsession
poetry run pytest tests
```
