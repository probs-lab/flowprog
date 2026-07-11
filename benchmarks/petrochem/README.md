# Petrochemicals model benchmark

A self-contained copy of the model behind the [C-THRU Global Petrochemicals
Calculator](https://github.com/c-thru/global-petrochemicals-calculator),
used here as a realistic, large, complex test case for flowprog development.

In the calculator repo, the model structure (which processes exist, what they
produce/consume, and the recipe coefficients) is built from RDF via
`flowprog.load_from_rdf`. `model_data.json` here is a one-time export of the
result of that process, bypassing jupyter-book etc, so everything below only
needs `flowprog` + `sympy`.

## Files

- `model_data.json` -- exported structure (143 processes, 111 objects, 552
  recipe coefficients), plus 21 reference scenarios (`params` -> expected
  `results`), plus two process-id lists (`processes_with_elec_req`,
  `processes_with_process_emissions`) used to group utility/emissions results.
- `structure.py` -- loads `model_data.json` and reconstructs
  `(ModelBuilder, recipe_data)`.
- `definitions.py` -- a few small object/process id lists used by
  `model_polymers.py`.
- `utils.py` -- small parameter-declaration and capacity-limit helpers.
- `model_polymers.py`, `model_fertilisers.py`, `model.py` -- the imperative
  model-building logic (`ModelBuilder.pull_production()` /
  `push_consumption()` / `limit()` / ... calls), copied over from the
  calculator's `load_model_polymers.py` / `load_model_fertilisers.py` /
  `load_model.py` essentially unchanged.
- `run_benchmark.py` -- builds the model, times each phase (structure load /
  build / compile / lambdify), then evaluates all 21 reference scenarios and
  checks the results match `model_data.json`'s `results`.

## Running

```
python benchmarks/petrochem/run_benchmark.py
```

## Regenerating model_data.json

If the calculator's model changes, re-run an export of the data (see
`example_export_standalone.py`) in the calculator repo and copy
`model/standalone_export.json` over `model_data.json` here, along with changes
to the model-building logic.
