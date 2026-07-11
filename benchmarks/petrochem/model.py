"""Top-level model assembly: combine the polymer and fertiliser sub-models."""

from model_polymers import define_polymer_model
from model_fertilisers import define_fertiliser_model


def define_model(
    model_builder, recipe_data, processes_with_elec_req, processes_with_process_emissions
):
    """Add all build steps to `model_builder` and return the dict of
    result expressions (to be passed to `SympyModel.lambdify(expressions=...)`)."""
    other_results_polymer = define_polymer_model(
        model_builder,
        recipe_data,
        processes_with_elec_req,
        processes_with_process_emissions,
    )
    other_results_fertiliser = define_fertiliser_model(model_builder)

    other_results = {**other_results_polymer, **other_results_fertiliser}
    other_results["GHG_biomass"] = other_results["GHG_biomass_nonfertiliser"]
    other_results["GHG_production"] = (
        other_results["GHG_production_nonfertiliser"]
        + other_results["GHG_production_fertiliser"]
    )
    other_results["GHG_use"] = other_results["GHG_use_fertiliser"]
    other_results["GHG_eol"] = other_results["GHG_eol_nonfertiliser"]
    other_results["GHG_total"] = (
        other_results["GHG_biomass"]
        + other_results["GHG_production"]
        + other_results["GHG_use"]
        + other_results["GHG_eol"]
    )

    return other_results
