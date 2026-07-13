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
- `regenerate_results.py` -- one-off migration utility that re-baselined
  `model_data.json`'s reference `results` for the elementary-exchanges
  migration (see below); kept for provenance, not part of the normal
  workflow.

## Running

```
python benchmarks/petrochem/run_benchmark.py
```

## Regenerating model_data.json

If the calculator's model changes, re-run an export of the data (see
`example_export_standalone.py`) in the calculator repo and copy
`model/standalone_export.json` over `model_data.json` here, along with changes
to the model-building logic.

## Elementary exchanges migration

`structure.py` declares elementary exchanges (`CO2`, `CH4`, `N2O`,
`GHG_upstream_Feedstock`, `GHG_upstream_Electricity`,
`GHG_ProcessHeat_Combustion`, `GHG_upstream_ProcessHeat`, `CO2_captured`) and
builds the full model structure: the 143 processes/111 objects from
`model_data.json`, plus 3
new objects (`Electricity`, `LowCarbonElectricity`, `ProcessHeat` -- energy
metric, not mass, so they don't show up in mass-flow Sankeys) and 12
generated `Source` boundary processes (via `flowprog.boundary_processes`):

- One per feedstock with no pre-existing supplying process (`NaturalGas`,
  `Coal`, `SugarCane`, `Maize`, `CornStover`, `SugarCaneBagasse`,
  `WheatStraw`, `RiceStraw`, `CapturedCarbonDioxide`), carrying a
  `GHG_upstream_Feedstock` B entry (`EF_Feedstock_<object>`). Naphtha/Ethane/
  Propane/Butane don't need one: they already have an explicit
  `OilRefining*` supplying process, which instead carries an *added* B entry
  via `add_direct_emission_and_feedstock_exchanges()`.
- `SourceOfElectricity` / `SourceOfLowCarbonElectricity`, each carrying a
  `GHG_upstream_Electricity` B entry (`EF_Utility_Electricity` and the fixed
  `0.007` respectively). `LowCarbonElectricity` is a distinct object/Source
  feeding only `WaterElectrolysisForHydrogen` (the green-hydrogen route) --
  the plan's own worked example (section 3.3) for "distinct qualities are
  distinct objects with distinct supply processes". Every other process in
  `processes_with_elec_req` consumes ordinary `Electricity`.
- `SourceOfProcessHeat`, carrying three B entries: `GHG_ProcessHeat_Combustion`
  (abated by `a_ccs_utility_combustion`), `GHG_upstream_ProcessHeat`
  (well-to-tank, unabated), and `CO2_captured` (the complementary
  `a_ccs_utility_combustion` fraction -- see below). Natural-gas-fired process
  heat is its own boundary-supplied object rather than mixed into
  `NaturalGas`-the-feedstock's technosphere flow, so combustion and WTT stay
  two distinct, separately reportable exchanges and `FeedstockInput_NaturalGas`
  keeps meaning feedstock use only.

All 71 processes in `processes_with_elec_req` get real `U[Electricity or
LowCarbonElectricity, j]` and `U[ProcessHeat, j]` technosphere entries (the
same `ElecReq_<process>` / `NGReq_<process>` symbols the ad-hoc calculation
already used). `calc_utility_requirements()` reads these straight off U via
`Reporting.consumption("Electricity"/"LowCarbonElectricity"/"ProcessHeat",
by="stage")` -- the technosphere analogue of `Reporting.aggregate()`'s
elementary-flow group-by -- instead of reconstructing the `ElecReq_<process>`
symbol names by hand. `model_polymers.py`'s `dispatch_boundary_processes()`
pulls the remaining production deficit of every boundary-supplied object
through its `Source` process (canonical pattern, plan section 3.2), closing
each of these 12 markets exactly -- `test_migration.py::
test_boundary_supplied_objects_balance_after_dispatch` checks this.

`calc_emissions()` is a pure consumer of one shared `Reporting` -- it takes no
model handle at all; every result is an aggregation over the (utility-
reattributed) elementary-flow table. Utility burdens sit on shared `Source`
processes rather than being distributed per-consumer, so the per-lifecycle-
stage breakdown goes through `flowprog.allocation.PassThrough`: each utility
Source's burden is reattributed to its direct consumers in proportion to
consumption (scope-limited allocation, closed-form and symbolic for these
input-less single-supplier Sources). Two stage pivots then do most of the
work:

- an uncharacterised `aggregate(None, by=("stage", "exchange"))` for the
  per-gas direct emissions (`Emissions_DirectProcess_{CO2,CH4,N2O}_*`) and the
  `CO2_captured` diagnostic that the `CCS` total reads back;
- a GWP-characterised `aggregate("GWPall", by=("stage", "source"))` -- a
  `source` grouping over exchange ids (`Elec`/`NGCombustion`/`NGWTT`/
  `Feedstock`/`Direct`) plus one `GWP_ALL` characterisation to kgCO2e -- whose
  columns are the per-stage source figures, whose column sums are
  `EmissionsBySource_*`, and whose row sums are `EmissionsByStage_*`. The old
  `add_sources`/`add_stages` prefix-scans over the results dict are gone.

Direct-emission processes need no explicit intersection: B-sparsity means only
processes carrying `DirProcEmis_*` entries have `CO2`/`CH4`/`N2O` cells, so the
`stage` grouping alone selects them. Feedstocks read straight off
`GHG_upstream_Feedstock` (producer-side, not reattributed): per group from a
`feedstock_group` grouping derived from the `FEEDSTOCKS` table, per object from
`aggregate(None, by=("process", "exchange"))`, and `FeedstockInput_*` from
`Reporting.production(object, limit_to_processes=[supplying process])` (the
limit is load-bearing -- `Naphtha` is also co-produced by chemical recycling).

CCS tracking follows the implementation plan's section 4 pattern directly:
every abated B entry (direct process emissions in
`add_direct_emission_and_feedstock_exchanges()`, process-heat combustion on
`SourceOfProcessHeat`) has a `CO2_captured` counterpart carrying the
complementary (captured) fraction, so the `CCS` result is just that one
exchange summed -- no parallel unabated calculation. Notably, the green-
hydrogen special case is gone from the reporting code:
`WaterElectrolysisForHydrogen` consumes `LowCarbonElectricity`, so the
reattribution routes the low-carbon EF to it structurally. The production-
deficit reporting shim is gone entirely (migration step 3) -- every feedstock
in `feedstock_emissions_params` now has an explicit `process_id`.

A performance note worth knowing before extending this further: eagerly
resolving expressions via `SympyModel.eval()` per quantity made an earlier
version of the benchmark's "build" phase regress from ~30s to 10+ minutes,
because the default `eval()` expands every intermediate in the model on every
call. Everything here therefore stays *raw* (intermediates unresolved) until
the model's single closing `lambdify(expressions=other_results)` call:
`SympyModel.eval(..., expand_intermediates=False)` resolves
`Y[j]*B[e,j]`/`X[j]*U[i,j]` role expressions to raw `Y[j]`/`X[j]` accumulated
values and recipe values (read directly via `get_recipe_as_symbols()`)
without expanding intermediates -- the same fix that was also needed inside
`flowprog.allocation` for `Allocation` to run on this model in ~10s rather
than hanging -- and `Reporting`/`PassThrough` build raw tables the same way
via `to_elementary_flows(raw=True)`. The remaining eager paths,
`Reporting.table()` and `to_elementary_flows()` without `raw=True`, still
have this characteristic at this model's scale.

### Deliberate reference-value changes

Migrating step 2 required re-baselining `model_data.json`'s 21 reference
scenarios (`regenerate_results.py`, kept for provenance). Every value not
listed below is bit-identical to the pre-migration ad-hoc calculation --
`regenerate_results.py` asserts this before writing the file, so it can't
silently launder an unrelated regression into the golden values. Three kinds
of change were made, all deliberate and reviewed:

1. **New keys**: `ProcessThroughput_SourceOf*` for the 12 new boundary
   processes (they now appear as flows, as the implementation plan
   anticipated).
2. **Fixed: `Emissions_ElecReq` (aggregate) used the wrong EF for the
   green-hydrogen slice.** Pre-migration, this aggregate was computed from
   the *total* electricity use times the ordinary `EF_Utility_Electricity`,
   silently applying it to the green-hydrogen slice too -- unlike the
   correctly-computed per-group breakdown (`Emissions_ElecReq_green_hydrogen`,
   which already used `0.007`). It's now the sum of the (already correct)
   per-group values, which the `LowCarbonElectricity` structural split makes
   fall out naturally rather than needing a special case in
   `calc_emissions()`. This is a real, material fix, not a rounding change:
   e.g. in the `only_green_hydrogen_capacity` scenario, `Emissions_ElecReq`
   drops from ~1126B to ~312B. `GHG_total` and every other headline total are
   unaffected -- they never depended on this particular aggregate key.
3. **Fixed: `EmissionsBySource_NGCombustion` / `EmissionsBySource_NGWTT` were
   exactly 2x too high**, in every scenario. Pre-migration, `calc_emissions()`
   looped over both the real per-group utility requirements *and* their own
   aggregate as if it were just another group, mislabelling it
   `Emissions_NGCombustion_NGReq` / `Emissions_NGWTT_NGReq`; the
   `EmissionsBySource_*` roll-up then summed the aggregate in alongside the
   real per-group values it was itself the sum of. The same double-counting
   also inflated the `CCS` diagnostic total wherever
   `a_ccs_utility_combustion` was nonzero (visible in the
   `only_ccs_utility_combustion` scenario). Fixed by summing only the real
   groups and renaming the aggregate keys to `Emissions_NGCombustion` /
   `Emissions_NGWTT` (matching the `Emissions_ElecReq` naming convention).
   `Emissions_NGReq` (a differently-shaped aggregate, unaffected by this bug)
   and `EmissionsBySource_NG` were already correct and are unchanged.
