# Changelog

## Unreleased

### Added

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

- `ModelStructure.expr()`'s multi-process roles (`SoldProduction`/
  `Consumption`/`ElementaryFlows`) now always return a sympy expression;
  previously an empty process list (e.g. `limit_to_processes` matching
  nothing) fell through Python's builtin `sum()` with no start value and
  returned a plain `int 0`.
- Serialisation format bumped to `"1.2"` (`ModelBuilder.save`/`load`) and
  `"1.1"` (`SympyModel.save`/`load`) to include elementary exchange
  declarations and B recipe entries; older files still load (with a
  version-mismatch warning).
