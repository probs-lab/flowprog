import sympy as sy

Z_fertiliser = sy.IndexedBase("Z_fertiliser")
E_fertiliser_use = sy.IndexedBase("E_fertiliser_use")
E_fertiliser_prod = sy.IndexedBase("E_fertiliser_prod")

fertiliser_names = [
    "AmmoniaFertiliser",
    "AmmoniumSulphate",
    "AmmoniumNitrate",
    "CalciumAmmoniumNitrate",
    "AmmoniumPhosphate",
    "NKCompound",
    "NPKCompound",
    "UreaAmmoniumNitrate",
    "OtherFertiliserN",
    "OtherFertiliserNP",
    "Urea",
]


def define_fertiliser_model(model):
    for i, name in enumerate(fertiliser_names):
        use_process = (
            f"UseOf{name}Fertiliser" if "Fertiliser" not in name else f"UseOf{name}"
        )
        # Z_fertiliser is measured in ktN. FIXME: if future changes include
        # showing fertiliser flows in the same Sankey diagram as the other
        # chemical flows, this will need to be converted to consistent units.
        # The emissions calculations below are correct.
        model.add(
            model.pull_production(name, Z_fertiliser[i]),
            model.push_consumption(name, Z_fertiliser[i]),
            label="Fertiliser demand",
        )

    # Parameters are in ktN and ktCO2e/ktN. Convert to kgCO2e for consistency
    # with the rest of the model.
    other_results = {
        "GHG_use_fertiliser": sum(
            Z_fertiliser[i] * E_fertiliser_use[i] * 1e6
            for i in range(len(fertiliser_names))
        ),
        "GHG_production_fertiliser": sum(
            Z_fertiliser[i] * E_fertiliser_prod[i] * 1e6
            for i in range(len(fertiliser_names))
        ),
        "GHG_eol_fertiliser": sy.S.Zero,
    }
    other_results["GHG_fertiliser"] = (
        other_results["GHG_use_fertiliser"] + other_results["GHG_production_fertiliser"]
    )

    return other_results
