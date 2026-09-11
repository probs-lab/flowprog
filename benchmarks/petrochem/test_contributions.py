"""Contribution analysis on the petrochemical baseline.

Breaks down the burden of a unit of vinyl chloride with several different sets
of processes expanded, and checks each breakdown accounts for the whole of it.
"""

import pytest

from structure import load_data, build_structure
from model import define_model
from flowprog.allocation import AllocatedSystem, ByValue, Excluding, ProcessSet

# Outputs that should take no share of the burden of the process emitting them.
RESIDUAL_OUTPUTS = {
    "Air",
    "MiscRefineryProducts",
    "Nutrients",
    "PyrolysisResidue",
    "Waste",
    "WasteBiomass",
    "WasteOtherChemicals",
    "Water",
}

UTILITIES = ["Electricity", "LowCarbonElectricity", "ProcessHeat"]


@pytest.fixture(scope="module")
def allocation():
    data = load_data()
    model_builder, recipe_data = build_structure(data)
    define_model(
        model_builder, recipe_data, data["processes_with_process_emissions"]
    )
    model = model_builder.build(recipe_data)
    params = next(iter(data["scenarios"].values()))["params"]
    rule = Excluding(RESIDUAL_OUTPUTS, ByValue())
    return model, AllocatedSystem(model, params, rule)


def test_first_tier_reports_the_producing_process_and_its_inputs(allocation):
    model, result = allocation
    expand = ProcessSet.producers_of(model.structure, "VinylChloride")
    assert expand.processes == {"VinylChlorideSynthesis"}

    breakdown = result.contributions(object="VinylChloride", expand=expand)
    assert breakdown.check(atol=1e-6)

    inputs = set(breakdown.upstream.index.get_level_values("object"))
    assert inputs <= set(model.structure.processes[
        model.structure.lookup_process("VinylChlorideSynthesis")
    ].consumes)


def test_inputs_can_be_reported_in_categories(allocation):
    model, result = allocation
    expand = ProcessSet.producers_of(model.structure, "VinylChloride")
    breakdown = result.contributions(object="VinylChloride", expand=expand).with_group(
        "category",
        {
            "chlorine": ["Chlorine"],
            "other intermediates": ["Ethylene", "InorganicAcids", "PureOxygen"],
            "electricity": ["Electricity", "LowCarbonElectricity"],
            "gas": ["ProcessHeat", "NaturalGas"],
        },
        direct_label="direct process emissions",
    )
    by_category = breakdown.by("category")
    assert breakdown.check(atol=1e-6)
    # Every slice is accounted for by the categories.
    assert by_category.sum().to_dict() == pytest.approx(breakdown.total.to_dict())


@pytest.mark.parametrize("tiers", [1, 2, 3])
def test_widening_the_boundary_still_accounts_for_everything(allocation, tiers):
    model, result = allocation
    expand = ProcessSet.tiers(model.structure, tiers, object="VinylChloride")
    breakdown = result.contributions(object="VinylChloride", expand=expand)
    assert breakdown.check(atol=1e-6)


def test_stopping_at_the_utilities(allocation):
    model, result = allocation
    expand = ProcessSet.upstream_of(
        model.structure, object="VinylChloride", stopping_at_objects=UTILITIES
    )
    breakdown = result.contributions(object="VinylChloride", expand=expand)
    assert breakdown.check(atol=1e-6)

    # Utility burdens are collapsed rather than attributed to the processes
    # supplying them.
    crossing = set(breakdown.upstream.index.get_level_values("object"))
    assert crossing & set(UTILITIES)
    assert not (expand.processes & set(model.structure.producers_of("Electricity")))


def test_a_process_reference_breaks_down_its_own_intensity(allocation):
    model, result = allocation
    expand = ProcessSet.of("VinylChlorideSynthesis")
    breakdown = result.contributions(process="VinylChlorideSynthesis", expand=expand)
    assert breakdown.check(atol=1e-6)
