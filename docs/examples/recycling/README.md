# Recycling example model

Run the notebook `recycling-model.ipynb`.

The model structure is read straight from `system-definitions.md` by
`load_model.py`. This needs the parser for system definitions, which `uv sync` installs along with the
rest of the development dependencies. To install it on its own:

``` shell
uv sync --extra definitions
```
