"""Tests for the elementary-exchange migration (flowprog implementation plan,
section 7, migration steps 1-3 -- all now complete; see README.md).

Checked here:
1. The golden regression: every value in model_data.json's 21 reference
   scenarios is reproduced exactly (run_benchmark.build()/verify()). The
   reference values were deliberately re-baselined as part of this migration
   for two documented bug fixes (see README.md) -- everything else matches
   the pre-migration ad-hoc calculation bit-for-bit.
2. Direct process emissions and every feedstock (not just the paraffins with
   a pre-existing supplying process) are B (elementary exchange) recipe
   entries, with no remaining production-deficit reporting shim.
3. Electricity/LowCarbonElectricity/ProcessHeat are real boundary-supplied
   technosphere objects, dispatched and market-balanced like any other.

Note: deliberately does NOT use SympyModel.eval()/to_elementary_flows() to
check this. Those eagerly substitute recipe/intermediate values via repeated
.subs() calls, which is fine for the small models flowprog's own test suite
uses but is far too slow at this model's scale (a single such call can take
seconds -- see SympyModel.eval(..., expand_intermediates=False), used
throughout model_polymers.py's calc_* functions instead).
Comparing the raw symbolic expressions directly (no substitution) is fast
and just as conclusive.
"""

import pytest
import sympy as sy

from run_benchmark import build, verify
from structure import (
    load_data,
    build_structure,
    FEEDSTOCK_BOUNDARY_OBJECTS,
    ELECTRICITY_EXCHANGE_ID,
    PROCESS_HEAT_COMBUSTION_EXCHANGE_ID,
    PROCESS_HEAT_WTT_EXCHANGE_ID,
    CAPTURED_CO2_EXCHANGE_ID,
    a_ccs_utility_combustion,
)
from model import define_model
from model_polymers import (
    GWP,
    FEEDSTOCK_EXCHANGE_ID,
    FEEDSTOCK_PROCESS_OBJECTS,
    BOUNDARY_SUPPLIED_OBJECTS,
    add_direct_emission_and_feedstock_exchanges,
)


def test_golden_regression():
    """Acceptance test 1: every quantity in model_data.json's results dict is
    reproduced exactly by the migrated model."""
    data, func = build()
    assert verify(data, func)


def test_direct_process_emissions_are_B_entries():
    """Direct process emissions (CO2/CH4/N2O, abated) are populated as B
    recipe entries with the expected symbolic formula."""
    data = load_data()
    model_builder, recipe_data = build_structure(data)

    exchanges = {e.id for e in model_builder.structure.elementary_exchanges}
    assert {"CO2", "CH4", "N2O"} <= exchanges

    processes_with_direct_emissions = data["processes_with_process_emissions"]
    add_direct_emission_and_feedstock_exchanges(
        model_builder, recipe_data, processes_with_direct_emissions
    )

    from model_polymers import abatement_for_process

    for process_id in processes_with_direct_emissions:
        j = model_builder._lookup_process(process_id)
        abatement = abatement_for_process(process_id)
        for ghg in GWP:
            e = model_builder.structure.lookup_exchange(ghg)
            expected = 1000 * sy.Symbol(f"DirProcEmis_{ghg}_{process_id}") * abatement
            assert recipe_data[model_builder.B[e, j]] == expected


def test_direct_process_emissions_have_captured_counterpart():
    """Every direct-process abated B entry has a CO2_captured counterpart
    carrying the complementary (captured) fraction -- implementation plan
    section 4's pattern, read by calc_emissions() as a single aggregate for
    the `CCS` result instead of a parallel unabated calculation."""
    data = load_data()
    model_builder, recipe_data = build_structure(data)

    exchanges = {e.id for e in model_builder.structure.elementary_exchanges}
    assert CAPTURED_CO2_EXCHANGE_ID in exchanges

    processes_with_direct_emissions = data["processes_with_process_emissions"]
    add_direct_emission_and_feedstock_exchanges(
        model_builder, recipe_data, processes_with_direct_emissions
    )

    from model_polymers import abatement_for_process

    e_captured = model_builder.structure.lookup_exchange(CAPTURED_CO2_EXCHANGE_ID)
    for process_id in processes_with_direct_emissions:
        j = model_builder._lookup_process(process_id)
        captured = 1 - abatement_for_process(process_id)
        expected = 1000 * captured * sum(
            sy.Symbol(f"DirProcEmis_{ghg}_{process_id}") for ghg in GWP
        )
        assert recipe_data[model_builder.B[e_captured, j]] == expected


def test_feedstock_emissions_with_explicit_process_are_B_entries():
    """Naphtha/Ethane/Propane/Butane feedstock emissions (which have an
    explicit supplying process already in the model) are B entries."""
    data = load_data()
    model_builder, recipe_data = build_structure(data)

    exchanges = {e.id for e in model_builder.structure.elementary_exchanges}
    assert FEEDSTOCK_EXCHANGE_ID in exchanges
    assert set(FEEDSTOCK_PROCESS_OBJECTS) == {
        "OilRefiningNaphtha",
        "OilRefiningEthane",
        "OilRefiningPropane",
        "OilRefiningButane",
    }

    add_direct_emission_and_feedstock_exchanges(
        model_builder, recipe_data, data["processes_with_process_emissions"]
    )
    e = model_builder.structure.lookup_exchange(FEEDSTOCK_EXCHANGE_ID)

    for process_id, object_id in FEEDSTOCK_PROCESS_OBJECTS.items():
        j = model_builder._lookup_process(process_id)
        assert object_id in model_builder.processes[j].produces
        assert recipe_data[model_builder.B[e, j]] == sy.Symbol(
            f"EF_Feedstock_{object_id}"
        )


def test_elementary_flows_expr_sums_Y_times_B_over_group():
    """expr("ElementaryFlows", exchange_id=e, limit_to_processes=group)
    constructs exactly sum(Y[j]*B[e,j] for j in group) -- the per-group,
    per-exchange direct-emission sum that calc_emissions() now obtains via
    the reporting layer's (stage, exchange) view over the elementary-flow table.
    (The un-expanded eval() resolution of such an expression is covered in
    tests/test_elementary_exchanges.py.)"""
    data = load_data()
    model_builder, recipe_data = build_structure(data)
    add_direct_emission_and_feedstock_exchanges(
        model_builder, recipe_data, data["processes_with_process_emissions"]
    )
    compiled = model_builder.build(recipe_data)

    group_process_ids = data["processes_with_process_emissions"][:2]
    result = compiled.expr(
        "ElementaryFlows", exchange_id="CO2", limit_to_processes=group_process_ids
    )

    e = compiled.structure.lookup_exchange("CO2")
    expected = sum(
        (
            compiled.Y[compiled.structure.lookup_process(pid)]
            * compiled.B[e, compiled.structure.lookup_process(pid)]
            for pid in group_process_ids
        ),
        sy.S.Zero,
    )
    assert result == expected


def test_every_feedstock_has_an_explicit_process_and_B_entry():
    """Migration step 2/3: every feedstock in feedstock_emissions_params has
    an explicit process_id (no more shim entries with process_id=None), and
    every one of them -- not just the paraffins -- has produced a B entry by
    the time the model is built."""
    from model_polymers import feedstock_emissions_params

    assert all(process_id is not None for _, _, process_id in feedstock_emissions_params)
    assert {object_id for _, object_id, _ in feedstock_emissions_params} >= set(
        FEEDSTOCK_BOUNDARY_OBJECTS
    )

    data = load_data()
    model_builder, recipe_data = build_structure(data)
    add_direct_emission_and_feedstock_exchanges(
        model_builder, recipe_data, data["processes_with_process_emissions"]
    )
    e = model_builder.structure.lookup_exchange(FEEDSTOCK_EXCHANGE_ID)
    for _, object_id, process_id in feedstock_emissions_params:
        j = model_builder._lookup_process(process_id)
        assert object_id in model_builder.processes[j].produces
        assert model_builder.B[e, j] in recipe_data


def test_boundary_objects_have_source_processes_with_B_entries():
    """Every object in structure.py's FEEDSTOCK_BOUNDARY_OBJECTS plus the
    3 utility objects gets a generated SourceOf* process carrying the
    expected B exchange(s), per structure.py's build_structure()."""
    data = load_data()
    model_builder, recipe_data = build_structure(data)

    e_feedstock = model_builder.structure.lookup_exchange(FEEDSTOCK_EXCHANGE_ID)
    for object_id in FEEDSTOCK_BOUNDARY_OBJECTS:
        j = model_builder._lookup_process(f"SourceOf{object_id}")
        assert object_id in model_builder.processes[j].produces
        assert recipe_data[model_builder.B[e_feedstock, j]] == sy.Symbol(
            f"EF_Feedstock_{object_id}"
        )

    e_elec = model_builder.structure.lookup_exchange(ELECTRICITY_EXCHANGE_ID)
    j_elec = model_builder._lookup_process("SourceOfElectricity")
    assert recipe_data[model_builder.B[e_elec, j_elec]] == sy.Symbol(
        "EF_Utility_Electricity"
    )
    j_elec_lc = model_builder._lookup_process("SourceOfLowCarbonElectricity")
    assert recipe_data[model_builder.B[e_elec, j_elec_lc]] == sy.Float(0.007)

    j_heat = model_builder._lookup_process("SourceOfProcessHeat")
    e_combustion = model_builder.structure.lookup_exchange(
        PROCESS_HEAT_COMBUSTION_EXCHANGE_ID
    )
    e_wtt = model_builder.structure.lookup_exchange(PROCESS_HEAT_WTT_EXCHANGE_ID)
    assert model_builder.B[e_combustion, j_heat] in recipe_data
    assert model_builder.B[e_wtt, j_heat] in recipe_data

    e_captured = model_builder.structure.lookup_exchange(CAPTURED_CO2_EXCHANGE_ID)
    assert recipe_data[model_builder.B[e_captured, j_heat]] == (
        sy.Symbol("EF_Utility_NaturalGas") * a_ccs_utility_combustion
    )


def test_green_hydrogen_consumes_low_carbon_electricity():
    """WaterElectrolysisForHydrogen (the sole green_hydrogen process) draws
    from LowCarbonElectricity, not ordinary Electricity -- the structural
    fix for the pre-migration aggregate-EF quirk (see README.md)."""
    data = load_data()
    model_builder, _ = build_structure(data)
    j = model_builder._lookup_process("WaterElectrolysisForHydrogen")
    process = model_builder.processes[j]
    assert "LowCarbonElectricity" in process.consumes
    assert "Electricity" not in process.consumes


def test_boundary_supplied_objects_balance_after_dispatch():
    """Every boundary-supplied object (feedstocks + utilities) has zero
    production deficit once the full model is built --
    dispatch_boundary_processes() closes each of these markets exactly, with
    no shim left reading a deficit for reporting purposes.

    Uses resolve_structural_symbols() + a single lambdify() call rather than
    repeated SympyModel.eval() -- see the module docstring for why eager
    per-call substitution doesn't scale at this model's size.
    """
    data = load_data()
    model_builder, recipe_data = build_structure(data)
    define_model(
        model_builder,
        recipe_data,
        data["processes_with_process_emissions"],
    )
    model = model_builder.build(recipe_data)

    deficit_exprs = {
        object_id: model.structure.resolve_structural_symbols(
            model.structure.ProductionDeficit[model.structure.lookup_object(object_id)],
            model._values,
        )
        for object_id in BOUNDARY_SUPPLIED_OBJECTS
    }
    func = model.lambdify(expressions=deficit_exprs, modules="math")

    scenario = next(iter(data["scenarios"].values()))
    deficits = func(scenario["params"])
    for object_id in BOUNDARY_SUPPLIED_OBJECTS:
        assert float(deficits[object_id]) == pytest.approx(0, abs=1e-6), object_id
