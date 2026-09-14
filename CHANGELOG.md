# Changelog

## Unreleased

### Added

- **Contribution analysis**: `AllocatedSystem.contributions()` breaks the
  burden carried by an object (or a process, or an object from a particular
  process) into named parts, showing direct and upstream emissions.

- **Market balance checking** (`flowprog.balance`): compiling a model now tracks
  whether each object with `has_market=True` actually balances, and the result is
  available as `model.balance_trace`. A market is reported as *balanced* only if
  production equals consumption for every parameter value; otherwise it is
  *conditional* (balances in some parameter regimes) or *open* (cannot balance at
  all). `BalanceTrace.explain(object_id)` gives a step-by-step account, and
  `BalanceTrace.breakpoints(object_id)` the conditions a conditional market depends
  on. A warning is logged after building a model that does not balance
  everywhere.

- **Elementary exchanges**: new support for elementary exchanges (in the LCA
  sense). Exchanges are represented by a `B[e, j]` signed coefficient matrix,
  declared via `ElementaryExchange` in the model structure. Exchange totals can
  be retrieved by `SympyModel.to_elementary_flows()` (the elementary-exchange
  analogue of `to_flows()`).
- **Lazy flow expressions**: `SympyModel.to_elementary_flows(raw=True)` returns
  raw symbolic values (accumulated Y expression x recipe B value, intermediate
  placeholders unresolved) instead of eagerly `eval()`-ing per row. This is much
  faster for large models, and the resulting expressions can be resolved in one
  pass via `lambdify(expressions=...)` or `eval()`. `SympyModel.eval()` gains an
  `expand_intermediates` flag which works similarly.
- **Boundary processes** (`flowprog.boundary_processes`): `Import`/`Export`/
  `Source`/`Sink` specs provide a declarative way to define simple processes
  that supply or consume specific objects, with associated elementary exchanges.

### Changed

- Changed default value for `lambdify()` `modules` parameter to "math"; this is
  much faster and works better than the previous default "numpy", but does not
  support vectorised evaluation across arrays. Pass `modules="numpy"` to get the
  previous default behaviour.

- `Allocation` now holds the results directly rather than behind a `.result`
  attribute, and additional outputs are now available.

- Internal allocation changes to handle processes whose reference flow is an
  input, and with possible stock accumulation. Process intensities are no longer
  calculated for processes with zero activity.

- **Allocation rules** must now be expressed in a new structure. This is more
  robust because it ensures all of a process's outputs are handled at once. The
  new rule names are `Mass`, `ByProperty`, `Fixed`, `Excluding` (to allocate no
  burden to certain objects) and `Rules` (a container for default allocation
  rules with per-process overrides). `Allocation` now has a `wastes=` shorthand
  to simultaneously set objects as cutoff (no burden on input to processes) and
  with zero allocation.

- **Resolving placeholder symbols moved off `ModelStructure`.**
  `ModelStructure.resolve_structural_symbols()` has been removed. Use
  `SympyModel.resolve(expr)`, or just pass expressions to `eval()`/`lambdify()`,
  which resolve them for you.

- **Compiling is now `SympyCompiler`** (in `flowprog.backends.sympy`), which
  walks the steps and accumulates activities and object balances;
  `SympyModel.from_steps()` is a thin wrapper over it. `SympyModel.__init__`
  takes the resulting `balance_trace` alongside `values` and `intermediates`,
  and it is saved with the model (format 1.3, under `object_balances`), so a
  model saved and loaded again can still say which of its markets balance. A
  model built without that history works the balances out from its activities
  instead, which gives the same quantities in a form nothing can be proved
  from.

- `ModelStructure.expr()`'s multi-process roles (`SoldProduction`/
  `Consumption`/`ElementaryFlows`) now always return a sympy expression;
  previously an empty process list (e.g. `limit_to_processes` matching
  nothing) fell through Python's builtin `sum()` with no start value and
  returned a plain `int 0`.

- Serialisation format bumped to `"1.2"` (`ModelBuilder.save`/`load`) and
  `"1.1"` (`SympyModel.save`/`load`) to include elementary exchange
  declarations and B recipe entries; older files still load (with a
  version-mismatch warning).
