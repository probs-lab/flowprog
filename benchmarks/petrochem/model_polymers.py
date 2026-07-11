import sympy as sy
from flowprog import merge_activities

from utils import (
    def_scalar_param,
    def_vector_param,
    pull_production_with_capacity_limit,
)

######### DEFINE PARAMS ############

Z_product = def_vector_param("Z_product")  # units: t
Z_EOL = def_vector_param("Z_EOL")
RR_C = def_vector_param("RR_C")
RR_M = def_vector_param("RR_M")
FT = def_vector_param("FT")

# Capacity for various production routes
C_ethyl_alcohol_from_biomass = def_scalar_param("C_ethyl_alcohol_from_biomass")
C_syngas_from_biomass = def_scalar_param("C_syngas_from_biomass")
C_ethylene_from_methyl_alcohol = def_scalar_param("C_ethylene_from_methyl_alcohol")
C_xylenes_from_methyl_alcohol = def_scalar_param("C_xylenes_from_methyl_alcohol")
C_green_hydrogen = def_scalar_param("C_green_hydrogen")
C_blue_hydrogen = def_scalar_param("C_blue_hydrogen")

# Fraction of biomass from different sources for ethyl alcohol production
k_ethyl_alcohol_biomass_feedstock_fraction = def_vector_param(
    "k_ethyl_alcohol_biomass_feedstock_fraction"
)
k_syngas_biomass_feedstock_fraction = def_vector_param(
    "k_syngas_biomass_feedstock_fraction"
)

# Fraction of olefin fossil feedstocks from ethane (as opposed to naphtha)
k_olefins_from_paraffins_ethane_fraction = def_scalar_param(
    "k_olefins_from_paraffins_ethane_fraction"
)

# Extra demand for chemicals not explicitly modelled as polymers
Z_extra = def_vector_param("Z_extra")

extra_demand_names = [
    "Benzene",
    "Toluene",
    "Xylenes",
    "MethylAlcohol",
]

# CCS deployment rates -- what fraction of emissions are avoided in each category
#
a_ccs_utility_combustion = def_scalar_param("a_ccs_utility_combustion")
a_ccs_process_emissions = def_scalar_param(
    "a_ccs_process_emissions"
)  # excluding incineration
a_ccs_incineration = def_scalar_param("a_ccs_incineration")

#####################################

from definitions import (
    organic_chemicals_synthesis,
    polymerisation_processes,
    polymer_objects,
)

product_names = [
    "PackagingProducts",
    "TransportationProducts",
    "BuildingsAndConstructionProducts",
    "ElectricalAndElectronicProducts",
    "ConsumerAndInstitutionalProducts",
    "IndustrialMachinery",
    "TextileProducts",
    "OtherProducts",
]

eol_polymers = [f"{name}AtEOL" for name in polymer_objects]

primary_chemicals = [
    "Ethylene",
    "Propylene",
    "Butadiene",
    "Benzene",
    "Butylenes",
    "Toluene",
    "Xylenes",
]

# FIXME: treat NaturalGas and Coal the same?
paraffins = ["Naphtha", "Ethane", "Propane", "Butane"]

# Parameters for calculating feedstock emissions.
# 3 fields: group, object_id, [optional] process_id
feedstock_emissions_params = [
    ("fossil", "NaturalGas", None),
    ("fossil", "Coal", None),
    ("fossil", "Naphtha", "OilRefiningNaphtha"),
    ("fossil", "Ethane", "OilRefiningEthane"),
    ("fossil", "Propane", "OilRefiningPropane"),
    ("fossil", "Butane", "OilRefiningButane"),
    ("biomass", "SugarCane", None),
    ("biomass", "Maize", None),
    ("biomass", "CornStover", None),
    ("biomass", "SugarCaneBagasse", None),
    ("biomass", "WheatStraw", None),
    ("biomass", "RiceStraw", None),
    ("co2", "CapturedCarbonDioxide", None),
]

##
## These functions define the individual steps of the logic
##


def ethylene_from_biomass(model):
    """Try to meet ethylene demand via ethyl alcohol, up to capacity limit."""
    ethyl_alcohol_production_processes = [
        "EthylAlcoholSynthesisFromCornStover",
        "EthylAlcoholSynthesisFromMaize",
        "EthylAlcoholSynthesisFromSugarcane",
        "EthylAlcoholSynthesisFromSugarcaneBagasse",
        "EthylAlcoholSynthesisFromWheatStraw",
        "EthylAlcoholSynthesisFromRiceStraw",
    ]
    model.add(
        pull_production_with_capacity_limit(
            model,
            "Ethylene",
            allocate_backwards={
                "Ethylene": {
                    "DehydrationOfEthylAlcohol": 1.0,
                },
                "EthylAlcohol": {
                    k: k_ethyl_alcohol_biomass_feedstock_fraction[i]
                    for i, k in enumerate(ethyl_alcohol_production_processes)
                },
            },
            # some natural gas is consumed via ammonia and syngas
            until_objects=["Syngas"],
            # Capacity limit for this route is for ethyl alcohol output
            limit_object="EthylAlcohol",
            limit_processes=ethyl_alcohol_production_processes,
            capacity=C_ethyl_alcohol_from_biomass,
        ),
        label="Production of ethylene from biomass",
    )


def ethylene_from_methyl_alcohol(model):
    """Try to meet ethylene demand via methyl alcohol, up to capacity limit."""
    # Only go as far back as methyl alcohol; its production is resolved in a
    # later step.

    # Initial proposal for required production, before considering the capacity limit
    proposal = model.pull_production(
        "Ethylene",
        model.object_production_deficit("Ethylene"),
        allocate_backwards={
            "Ethylene": {
                "MethylAlcoholToOlefins": 1,
            },
        },
        until_objects=["NaturalGas", "MethylAlcohol", "PureHydrogen"],
    )

    # Limit by capacity for MTO, defined slightly loosely as "total ethylene or propylene"
    limited = model.limit(
        proposal,
        (
            model.expr(
                "ProcessOutput",
                process_id="MethylAlcoholToOlefins",
                object_id="Ethylene",
            )
            + model.expr(
                "ProcessOutput",
                process_id="MethylAlcoholToOlefins",
                object_id="Propylene",
            )
        ),
        limit=C_ethylene_from_methyl_alcohol,
    )
    model.add(
        limited,
        label="Production of ethylene from methyl alcohol",
    )


def ethylene_from_paraffins(model):
    """
    Produce ethylene from ethane/naphtha, up to a capacity limit

    Depending on where you are in the world, the main fossil feedstock is
    either ethane (USA -- with gas) or Naphtha (everywhere else) -- roughly speaking.

    So, split demand for ethylene/propylene into these different regions.
    """

    model.add(
        model.pull_production(
            "Ethylene",
            model.object_production_deficit("Ethylene"),
            allocate_backwards={
                "Ethylene": {
                    "SteamCrackingOfEthane": k_olefins_from_paraffins_ethane_fraction,
                    "SteamCrackingOfNaphtha": (
                        1 - k_olefins_from_paraffins_ethane_fraction
                    ),
                    "DehydrationOfEthylAlcohol": 0,
                    "FischerTropschSynthesisOfOlefinsFromSyngas": 0,
                    "FluidCatalyticCrackingOfGasOil": 0,
                    "MethylAlcoholToOlefins": 0,
                    "MethylAlcoholToPropylene": 0,
                },
                # This route is about oil-based ethane, don't go round to consider the
                # ethane that comes from steam cracking of methyl alcohol? XXX
                "Ethane": {"OilRefiningEthane": 1},
            },
            until_objects=["NaturalGas"],
        ),
        label="Production of ethylene from ethane/naphtha",
    )


def btx_from_methyl_alcohol(model):
    """Try to meet xylene demand via methyl alcohol, up to capacity limit."""
    # Currently xylene is the biggest of BTX by demand, so start by trying to
    # satisfy demand for that. Only go back as far as methyl alcohol; its
    # production is resolved in a separate step later.
    model.add(
        pull_production_with_capacity_limit(
            model,
            "Xylenes",
            allocate_backwards={
                "Xylenes": {
                    "MethylAlcoholToAromatics": 1,
                },
            },
            until_objects=["NaturalGas", "MethylAlcohol", "PureHydrogen"],
            # Capacity limit for this route is for xylenes output
            limit_object="Xylenes",
            limit_processes=["MethylAlcoholToAromatics"],
            capacity=C_xylenes_from_methyl_alcohol,
        ),
        label="Production of BTX from methyl alcohol",
    )


def btx_from_spare_pyrolysis_gasoline(model):
    """Use up spare pyrolysis gasoline converting to BTX."""
    # Run distillation processes using the available gasoline, forwards through the
    # model only as far as the BTX (primary chemicals).
    model.add(
        model.push_consumption(
            "PyrolysisGasoline",
            model.object_consumption_deficit("PyrolysisGasoline"),
            until_objects=primary_chemicals,
        ),
        label="Distill the available pyrolysis gasoline to BTX",
    )


def btx_from_spare_toluene(model):
    """Use up spare toluene converting to xylenes."""
    model.add(
        model.push_process_input(
            "DisproportionationOfTolueneForXylenes",
            "Toluene",
            model.object_consumption_deficit("Toluene"),
            until_objects=primary_chemicals + ["NaturalGas"],
        ),
        label="Convert spare toluene to xylenes",
    )


def btx_from_naphtha(model):
    """
    Supply residual BTX demand via naphtha.

    There are multiple routes to produce the BTX. Xylenes are the biggest
    demand, so start with them. Xylenes can be produced directly from naphtha,
    or via toluene. Introduce a capacity limit for the direct reforming route so
    that we can fall back on going via toluene if needed.
    """

    # First meet remaining demand for xylenes via catalytic reforming of naphtha
    # -- there is plenty of capacity available for this so suitable as a
    # fallback.
    #
    # References for capacity: Baseline and reference projected capacities for
    # naphtha catalytic reforming were based on the data of OPEC (2020) and OPEC
    # (2022), respectively, with average xylenes content of 22.5% (Bender 2013).
    # "OPEC (2020). 2020 World Oil Outlook 2045.
    # https://www.opec.org/opec_web/static_files_project/media/downloads/publications/OPEC_WOO2020.pdf;
    # OPEC (2022). 2022 World Oil Outlook 2045.
    # https://www.opec.org/opec_web/static_files_project/media/downloads/WOO_2022.pdf;
    # and Bender (2013). Global Aromatics Supply - Today and Tomorrow.
    # https://www.osti.gov/etdeweb/servlets/purl/22176034."
    model.add(
        model.pull_production(
            "Xylenes",
            model.object_production_deficit("Xylenes"),
            allocate_backwards={
                "Xylenes": {
                    "CatalyticReformingOfNaphthaForXylenes": 1,
                },
            },
            until_objects=["NaturalGas", "Syngas", "PureHydrogen"],
        ),
        label="Production of Xylenes (and other BTX) from naphtha",
    )

    # Now toluene (based on direct demand for toluene and for xylene -- keep
    # benzene til last since it is mixed together with toluene + benzene, and
    # otherwise we end up with too much benzene)
    model.add(
        model.pull_production(
            "Toluene",
            model.object_production_deficit("Toluene"),
            allocate_backwards={
                "Toluene": {
                    "CatalyticReformingOfNaphthaForToluene": 1,
                },
            },
            until_objects=["NaturalGas", "Syngas", "PureHydrogen"],
        ),
        label="Production of Toluene (and other BTX) from naphtha",
    )

    # Now make sure there is enough benzene (which will create additional demand
    # for toluene, and as a byproduct additional benzene -- so scale down)
    #
    # TODO Should get these values from the recipe data, not hard-coded here as
    # magic numbers!
    extra_benzene_factor = 1 + 1.187557466 * 0.239424947
    required_extra_demand = (
        model.object_production_deficit("Benzene") / extra_benzene_factor
    )
    model.add(
        model.pull_production(
            "Benzene",
            required_extra_demand,
            allocate_backwards={
                "Benzene": {
                    "DealkylationOfTolueneForBenzene": 1,
                },
                "Toluene": {
                    "CatalyticReformingOfNaphthaForToluene": 1,
                },
            },
            until_objects=["NaturalGas", "Syngas", "PureHydrogen"],
        ),
        label="Production of Benzene from toluene",
    )


def on_purpose_propylene(model):
    """On-purpose production of propylene.

    1. If we have spare capacity for MTO (after producing ethylene already),
    then use it for propylene.

    2. Use propane dehydrogenation for any remaining demand.

    """

    # Only go as far back as methyl alcohol; its production is resolved in a
    # later step.
    #
    # Note: this process does produce some ethylene, which has already been
    # balanced, but only approx 5%. So as a simplification we do not attempt to
    # get the right blend of MTP/MTO to match ethylene demand exactly.

    # Initial proposal for required production, before considering the capacity limit
    proposal = model.pull_production(
        "Propylene",
        model.object_production_deficit("Propylene"),
        allocate_backwards={
            "Propylene": {
                "MethylAlcoholToPropylene": 1,
            },
        },
        until_objects=["NaturalGas", "MethylAlcohol", "PureHydrogen"],
    )

    # Limit by capacity for MTO, defined slightly loosely as "total ethylene or propylene"
    limited = model.limit(
        proposal,
        (
            model.expr(
                "ProcessOutput",
                process_id="MethylAlcoholToOlefins",
                object_id="Ethylene",
            )
            + model.expr(
                "ProcessOutput",
                process_id="MethylAlcoholToOlefins",
                object_id="Propylene",
            )
            + model.expr(
                "ProcessOutput",
                process_id="MethylAlcoholToPropylene",
                object_id="Propylene",
            )
        ),
        limit=C_ethylene_from_methyl_alcohol,
    )
    model.add(
        limited,
        label="Production of propylene from methyl alcohol",
    )

    # Back-up on-purpose production
    model.add(
        model.pull_process_output(
            "DehydrogenationOfPropane",
            "Propylene",
            model.object_production_deficit("Propylene"),
            allocate_backwards={
                "Propane": {
                    "MethylAlcoholToOlefins": 0,
                    "OilRefiningPropane": 1,
                }
            },
        ),
        label="On-purpose production of propylene from propane",
    )


def on_purpose_butadiene(model):
    """On-purpose production of butadiene."""

    # First, there may be some butylenes available as by-products from steam
    # cracking, which are not being used yet.
    potential_distillation = model.pull_process_output(
        "DistillationOfButylenesForButadiene",
        "Butadiene",
        model.object_production_deficit("Butadiene"),
        until_objects=["Butylenes"],
    )
    model.add(
        # Only use the butylenes that are spare
        model.limit(
            potential_distillation,
            expr=model.expr(
                "ProcessInput",
                process_id="DistillationOfButylenesForButadiene",
                object_id="Butylenes",
            ),
            limit=model.object_consumption_deficit("Butylenes"),
        ),
        label="On-purpose production of butadiene from available butylenes",
    )

    # If we still need more butadiene, the next option is dehydrogenation of butane.
    model.add(
        model.pull_process_output(
            "DehydrogenationOfButaneForButadiene",
            "Butadiene",
            model.object_production_deficit("Butadiene"),
            # until_objects=["Butylenes"],
        ),
        label="On-purpose production of butadiene from butane",
    )


def methyl_alcohol_production(model):
    """Produce methyl alcohol demand from hydrogen or syngas.

    Green hydrogen is produced up to the capacity limit, then fall back to
    syngas for the rest.
    """

    # Production of methyl alcohol -- first from green hydrogen (subject to
    # capacity limit), then from blue hydrogen (subject to capacity limit)
    model.add(
        pull_production_with_capacity_limit(
            model,
            "MethylAlcohol",
            allocate_backwards={
                "MethylAlcohol": {
                    "CarbonDioxideHydrogenationToMethylAlcohol": 1,
                },
                "PureHydrogen": {
                    "WaterElectrolysisForHydrogen": 1,
                },
            },
            limit_object="PureHydrogen",
            limit_processes=["WaterElectrolysisForHydrogen"],
            capacity=C_green_hydrogen,
        ),
        label="Production of methyl alcohol from green hydrogen",
    )
    model.add(
        pull_production_with_capacity_limit(
            model,
            "MethylAlcohol",
            allocate_backwards={
                "MethylAlcohol": {
                    "CarbonDioxideHydrogenationToMethylAlcohol": 1,
                },
                "PureHydrogen": {
                    "NaturalGasSteamMethaneReformingWithCCSToHydrogen": 1,
                },
            },
            until_objects=["NaturalGas"],
            limit_object="PureHydrogen",
            limit_processes=["NaturalGasSteamMethaneReformingWithCCSToHydrogen"],
            capacity=C_blue_hydrogen,
        ),
        label="Production of methyl alcohol from blue hydrogen",
    )

    # Then top up remaining methyl alcohol demand from syngas. We asssume there
    # is no point producing more methyl alcohol from hydrogen unless the
    # hydrogen is clean (green/blue).
    model.add(
        model.pull_production(
            "MethylAlcohol",
            model.object_production_deficit("MethylAlcohol"),
            allocate_backwards={
                "MethylAlcohol": {
                    "MethylAlcoholSynthesis": 1,
                },
            },
            until_objects=["Syngas"],
        ),
        label="Production of methyl alcohol from syngas",
    )

    # Production of syngas. Prefer biomass routes up to capacity limit.
    syngas_biomass_production_processes = [
        "CornStoverGasificationToSyngas",
        "SugarCaneBagasseGasificationToSyngas",
        "WheatStrawGasificationToSyngas",
        "RiceStrawGasificationToSyngas",
    ]
    biomass_fractions = {
        k: k_syngas_biomass_feedstock_fraction[i]
        for i, k in enumerate(syngas_biomass_production_processes)
    }
    model.add(
        pull_production_with_capacity_limit(
            model,
            "Syngas",
            allocate_backwards={
                "Syngas": {
                    **biomass_fractions,
                    "CoalGasificationToSyngas": 0,
                    "NaturalGasSteamMethaneReformingToSyngas": 0,
                },
            },
            until_objects=["NaturalGas"],
            # Capacity limit for this route is for syngas output
            limit_object="Syngas",
            limit_processes=syngas_biomass_production_processes,
            capacity=C_syngas_from_biomass,
        ),
        label="Production of syngas from biomass",
    )

    # Finally, produce any remaining demand from fossil fuels.
    model.add(
        model.pull_production(
            "Syngas",
            model.object_production_deficit("Syngas"),
            allocate_backwards={
                "Syngas": {
                    # Current baseline syngas supply is currently approximately
                    # equally from coal and natural gas
                    # (https://pubs.acs.org/doi/10.1021/acssuschemeng.2c05390).
                    # This could be adjusted by a lever in future.
                    "CoalGasificationToSyngas": 0.5,
                    "NaturalGasSteamMethaneReformingToSyngas": 0.5,
                },
            },
            until_objects=["NaturalGas"],
        )
    )


def hydrogen_production(model):
    """Produce remaining hydrogen from water electrolysis or SMR."""

    # Try to use any remaining green hydrogen capacity first
    model.add(
        pull_production_with_capacity_limit(
            model,
            "PureHydrogen",
            allocate_backwards={"PureHydrogen": {"WaterElectrolysisForHydrogen": 1}},
            limit_object="PureHydrogen",
            limit_processes=["WaterElectrolysisForHydrogen"],
            capacity=C_green_hydrogen,
        ),
        label="Production of green hydrogen for all other uses",
    )

    # Now try to use any blue hydrogen capacity
    model.add(
        pull_production_with_capacity_limit(
            model,
            "PureHydrogen",
            allocate_backwards={
                "PureHydrogen": {"NaturalGasSteamMethaneReformingWithCCSToHydrogen": 1}
            },
            until_objects=["NaturalGas"],
            limit_object="PureHydrogen",
            limit_processes=["NaturalGasSteamMethaneReformingWithCCSToHydrogen"],
            capacity=C_blue_hydrogen,
        ),
        label="Production of blue hydrogen for all other uses",
    )

    # Finally produce any remaining hydrogen demand as grey hydrogen
    model.add(
        model.pull_production(
            "PureHydrogen",
            model.object_production_deficit("PureHydrogen"),
            allocate_backwards={
                "PureHydrogen": {"NaturalGasSteamMethaneReformingToHydrogen": 1}
            },
            until_objects=["NaturalGas"],
        )
    )


def fossil_paraffins_production(model):
    """Produce remaining demand for paraffins from fossil sources."""

    for paraffin in paraffins:
        model.add(
            model.pull_production(
                paraffin,
                model.object_production_deficit(paraffin),
                allocate_backwards={paraffin: {f"OilRefining{paraffin}": 1}},
            ),
            label=f"Fossil production of {paraffin}",
        )


def stock_model_flows(model):
    """Add demand and EOL flows based on stock model."""
    for i, name in enumerate(product_names):
        # Production to meet demand
        model.add(
            model.pull_production(name, Z_product[i], until_objects=polymer_objects),
            label="Product demand",
        )

        # End of life flows, up to the EOL polymers
        model.add(
            model.push_consumption(name, Z_EOL[i], until_objects=eol_polymers),
            label="Product EOL",
        )


def eol_polymer_treatment(model):
    """Process EOL polymers through mechanical/chemical recycling or other
    end-of-life treatment."""

    # Send a fraction of each EOL polymer to mechanical recycling, chemical
    # recycling, incineration, landfill, or "mismanagement" based on the
    # fraction parameters.

    for i, eol_polymer in enumerate(eol_polymers):
        # eol_polymer is, for example, "HDPEPolyethyleneAtEOL"
        quantity = model.object_consumption_deficit(eol_polymer)

        # A few polymers (Polyurethane, SyntheticRubbers, OtherPolymers) have no
        # mechanical recycling process modelled -- Mixing is their only
        # consumer, so no allocation coefficients are needed (or accepted) for
        # them here.
        has_mechanical_recycling = (
            f"MechanicalRecyclingOf{eol_polymer}" in model.consumers_of(eol_polymer)
        )
        if has_mechanical_recycling:
            mechanical_first = {
                eol_polymer: {
                    f"MechanicalRecyclingOf{eol_polymer}": 1,
                    f"Mixing{eol_polymer}": 0,
                }
            }
            mixing_first = {
                eol_polymer: {
                    f"MechanicalRecyclingOf{eol_polymer}": 0,
                    f"Mixing{eol_polymer}": 1,
                }
            }
        else:
            mechanical_first = {}
            mixing_first = {}

        # The 5 shares below (mechanical/chemical/3x final treatment) all
        # divide up the *same* `quantity`. object_consumption_deficit()
        # returns a placeholder that the compiler resolves against
        # accumulated state at the point of each model.add() -- if these were
        # added as 5 separate steps, each later share would see a smaller
        # "remaining" deficit (since earlier shares' own consumption would
        # already be part of the accumulated state), shrinking them
        # incorrectly. Merging them into one AdditionalActivity before
        # calling model.add() makes them resolve atomically, against the same
        # snapshot, so the 5 shares add back up to the original `quantity`.
        mechanical_activity = model.push_consumption(
            eol_polymer,
            quantity * RR_M[i],
            allocate_forwards={
                **mechanical_first,
                # XXX this shouldn't be needed here as the value above is zero
                "MixedPolymersAtEOL": {
                    "ChemicalRecyclingOfMixedPolymersAtEOL": 1,
                    "Incineration": 0,
                    "Landfilling": 0,
                    "Mismanagement": 0,
                },
            },
            until_objects=polymer_objects,
        )
        # Chemical recycling -- first through mixing, then chemical recycling
        chemical_activity = model.push_consumption(
            eol_polymer,
            quantity * RR_C[i],
            allocate_forwards={
                **mixing_first,
                "MixedPolymersAtEOL": {
                    "ChemicalRecyclingOfMixedPolymersAtEOL": 1,
                    "Incineration": 0,
                    "Landfilling": 0,
                    "Mismanagement": 0,
                },
            },
        )
        # Final treatment options
        final_treatment_activities = []
        for j, treatment_label in enumerate(
            ["Incineration", "Landfilling", "Mismanagement"]
        ):
            destination = {
                "ChemicalRecyclingOfMixedPolymersAtEOL": 0,
                "Incineration": 0,
                "Landfilling": 0,
                "Mismanagement": 0,
            }
            destination[treatment_label] = 1
            final_treatment_activities.append(
                model.push_consumption(
                    eol_polymer,
                    quantity * (1 - RR_M[i] - RR_C[i]) * FT[j],
                    allocate_forwards={
                        **mixing_first,
                        "MixedPolymersAtEOL": destination,
                    },
                )
            )
        model.add(
            merge_activities(
                mechanical_activity, chemical_activity, *final_treatment_activities
            ),
            label=f"EOL treatment of {eol_polymer} (mechanical/chemical recycling and final treatment)",
        )


def polymer_from_primary_chemicals(model, object_id, amount=None):
    """Product polymer `object_id` back as far as primary chemicals.

    Stops at HydrogenCyanide since that is a byproduct of AcrylonitrileSynthesis
    -- any remaining demand for HydrogenCyanide should be balanced after
    producing all required polymers.

    """

    processes = model.producers_of(object_id)

    # Filter out the EOLProcessing processes, we want to use the other remaining
    # process that's available to balance production of each polymer.
    process_candidates = [
        p for p in processes if not p.startswith("MechanicalRecycling")
    ]
    assert len(process_candidates) == 1
    process_id = process_candidates[0]

    if amount is None:
        amount = model.object_production_deficit(object_id)

    model.add(
        model.pull_process_output(
            process_id,
            object_id,
            amount,
            until_objects=(
                primary_chemicals
                + [
                    "NaturalGas",
                    "Syngas",  # for ammonia production
                    "MethylAlcohol",
                    "PureHydrogen",
                    "HydrogenCyanide",
                ]
            ),
        ),
        label=f"Polymer production ({object_id}) from primary chemicals",
    )


def balance_intermediate_chemicals_from_primary_chemicals(model):
    """Produce intermediate chemicals from primary chemicals.

    Production of polymers stops at HydrogenCyanide since that is a byproduct of
    AcrylonitrileSynthesis -- here we now produce any remaining demand for
    HydrogenCyanide.

    """

    model.add(
        model.pull_process_output(
            "HydrogenCyanideSynthesis",
            "HydrogenCyanide",
            model.object_production_deficit("HydrogenCyanide"),
            until_objects=(
                primary_chemicals
                + [
                    "NaturalGas",
                    "Syngas",  # for ammonia production
                    "MethylAlcohol",
                    "PureHydrogen",
                ]
            ),
        ),
        label=f"HydrogenCyanide production from primary chemicals",
    )


def add_other_consumption(model):
    """Add demand for chemicals not driven by polymers.

    Stops at primary chemicals.
    """
    for i, name in enumerate(extra_demand_names):
        process = f"OtherConsumptionOf{name}"
        model.add(
            model.pull_process_output(
                process,
                object_id=None,
                value=Z_extra[i],
                until_objects=primary_chemicals + ["MethylAlcohol"],
            ),
            label=f"Additional non-polymer demand for {name}",
        )


PROCESS_GROUPS = {
    "green_hydrogen": {
        "WaterElectrolysisForHydrogen",
    },
    "other_hydrogen": {
        "NaturalGasSteamMethaneReformingToHydrogen",
        "NaturalGasSteamMethaneReformingWithCCSToHydrogen",
    },
    "biomass": {
        "WheatStrawGasificationToSyngas",
        "RiceStrawGasificationToSyngas",
        "CornStoverGasificationToSyngas",
        "SugarCaneBagasseGasificationToSyngas",
        "EthylAlcoholSynthesisFromSugarcane",
        "EthylAlcoholSynthesisFromMaize",
        "EthylAlcoholSynthesisFromSugarcaneBagasse",
        "EthylAlcoholSynthesisFromCornStover",
        "EthylAlcoholSynthesisFromWheatStraw",
        "EthylAlcoholSynthesisFromRiceStraw",
    },
    "primary_production": {
        "SyngasToAmmoniaProduction",
        "MethylAlcoholToPropylene",
        "FischerTropschSynthesisOfOlefinsFromSyngas",
        "NaturalGasSteamMethaneReformingToSyngas",
        "CoalGasificationToSyngas",
        "DisproportionationOfTolueneForXylenes",
        "CatalyticReformingOfNaphthaForXylenes",
        "MethylAlcoholSynthesis",
        "CatalyticReformingOfNaphthaForToluene",
        "MethylAlcoholToOlefins",
        "DistillationOfButylenesForButadiene",
        "SteamCrackingOfEthane",
        "DehydrogenationOfButaneForButadiene",
        "CarbonDioxideHydrogenationToMethylAlcohol",
        "DealkylationOfTolueneForBenzene",
        "SteamCrackingOfNaphtha",
        "DehydrationOfEthylAlcohol",
        "DistillationOfPyrolysisGasolineForBTX",
        "FluidCatalyticCrackingOfGasOil",
        "DehydrogenationOfPropane",
        "MethylAlcoholToAromatics",
        "SodiumChlorideElectrolysisForChlorine",
    },
    "organic_synthesis": set(organic_chemicals_synthesis),
    "downstream": set(polymerisation_processes),
    "end_of_life": {
        "Landfilling",
        "Mismanagement",
        "Incineration",
        "MechanicalRecyclingOfLLDPEAtEOL",
        "MechanicalRecyclingOfPVCPolyvinylChlorideAtEOL",
        "ChemicalRecyclingOfMixedPolymersAtEOL",
        "MechanicalRecyclingOfPSPolystyreneAtEOL",
        "MechanicalRecyclingOfLDPEPolyethyleneAtEOL",
        "MechanicalRecyclingOfPETPolyethyleneTerephthalatePolyestersAtEOL",
        "MechanicalRecyclingOfPPPolypropyleneAtEOL",
        "MechanicalRecyclingOfHDPEPolyethyleneAtEOL",
        "MechanicalRecyclingOfFibrePPAAtEOL",
    },
}


def group_intersection(process_ids):
    """Prepare process groups. Result is intersection of each process group with
    `process_ids`. An "other" group is added if needed."""

    s = set(process_ids)
    groups = {k: s & v for k, v in PROCESS_GROUPS.items()}
    for k, v in groups.items():
        s -= v
    groups["other"] = s
    return groups


def process_factor_sum(model, values, factor_prefix, process_ids, extra_factors=None):
    if extra_factors is None:
        extra_factors = {}
    return sy.S(
        sum(
            values[model.Y[model._lookup_process(k)]]
            * sy.Symbol(f"{factor_prefix}{k}")
            * extra_factors.get(k, sy.S.One)
            for k in process_ids
        )
    )


def calc_flow_summaries(model, values):
    """Calculate summary metrics based on mass flows.

    `values` is the accumulated symbolic X[j]/Y[j] state, from
    `model.build()._values`, at this point in the imperative build sequence.
    """

    # Total polymer production -- make sure to distinguish production from
    # recycling vs from new polymerisation.
    expr_virgin = sum(
        model.expr(
            "ProcessOutput", process_id=polymerisation_name, object_id=polymer_name
        )
        for polymer_name, polymerisation_name in zip(
            polymer_objects, polymerisation_processes
        )
    )
    expr_recycling = sum(
        model.expr(
            "ProcessOutput",
            process_id=f"MechanicalRecyclingOf{polymer_name}AtEOL",
            object_id=polymer_name,
        )
        for polymer_name in polymer_objects
        if polymer_name
        in {
            "LDPEPolyethylene",
            "LLDPE",
            "HDPEPolyethylene",
            "PPPolypropylene",
            "PSPolystyrene",
            "PVCPolyvinylChloride",
            "PETPolyethyleneTerephthalatePolyesters",
            "FibrePPA",
        }
    )

    def subs_values(expr):
        # This should be hidden inside the model code somewhere, not here
        return expr.subs(
            {model.X[j]: values[model.X[j]] for j in range(len(model.processes))}
        ).subs(
            {model.Y[j]: values[model.Y[j]] for j in range(len(model.processes))}
        )

    results = {
        "Production_polymers_virgin": subs_values(expr_virgin),
        "Production_polymers_recycled": subs_values(expr_recycling),
    }

    # Add in process capacity for every process
    for p in model.processes:
        j = model._lookup_process(p.id)
        expr = sy.Max(model.X[j], model.Y[j])
        results[f"ProcessThroughput_{p.id}"] = subs_values(expr)

    return results


def calc_utility_requirements(model, values, processes_with_elec_req):
    # Find the ones in the groups that are relevant for utility requirements
    # only -- intersection of groups and `processes_with_elec_req`.
    groups = group_intersection(processes_with_elec_req)
    if groups["other"]:
        print("'Other' group for utility use: ", groups["other"])

    other_results = {}
    for k, v in groups.items():
        other_results[f"ElecReq_{k}"] = process_factor_sum(model, values, "ElecReq_", v)
        other_results[f"NGReq_{k}"] = process_factor_sum(model, values, "NGReq_", v)

    other_results["ElecReq"] = sum(
        v for k, v in other_results.items() if k.startswith("ElecReq_")
    )
    other_results["NGReq"] = sum(
        v for k, v in other_results.items() if k.startswith("NGReq_")
    )

    return other_results


def calc_emissions(model, values, utility_reqs, processes_with_direct_emissions):

    results = {}

    # Track use of CCS abatement total
    results["CCS"] = sy.S.Zero

    # Emissions from utility use by lifecycle stage.
    #
    # Units: emissions factors for utilities are in "kgCO2e/kWh" and
    # "kgCO2e/MJ". Utility requirements are in kWh and MJ already. Emissions
    # should be reported in kgCO2e. So all works out neatly.
    #
    # To include the upstream (Scope 3 / WTT) emissions for the natural gas
    # burned as a utility, we convert the MJ utility into t by dividing by the
    # energy content 50.4 GJ/t, then multiply by the EF_Feedstock_NaturalGas
    # (423 kgCO2e/t)
    for k, v in list(utility_reqs.items()):
        if k.startswith("ElecReq"):
            # Special case for green hydrogen: always use green low-carbon electricity
            if k == "ElecReq_green_hydrogen":
                # Meys et al use 7gCO2e/kWh as their minimal wind-based "green"
                # electricity. This parameter is in kgCO2e/kWh.
                EF = 0.007
            else:
                EF = sy.Symbol("EF_Utility_Electricity")
            # e.g. Emissions_ElecReq_end_of_life
            results["Emissions_" + k] = v * EF
        if k.startswith("NGReq"):
            original_emissions = v * sy.Symbol("EF_Utility_NaturalGas")
            abated_emissions = original_emissions * (1 - a_ccs_utility_combustion)
            # Convert MJ to tonnes: gross CV of natural gas is 50.4 GJ/t (DEFRA data)
            wtt_emissions = v / 50400 * sy.Symbol("EF_Feedstock_NaturalGas")
            results["Emissions_NGCombustion_" + k] = abated_emissions
            results["Emissions_NGWTT_" + k] = wtt_emissions
            results["Emissions_" + k] = abated_emissions + wtt_emissions
            results["CCS"] += original_emissions - abated_emissions

    # Direct process emissions
    #
    # Find the ones in the groups that are relevant for utility requirements
    # only -- intersection of groups and `processes_with_direct_emissions`.
    groups_dp = group_intersection(processes_with_direct_emissions)
    if groups_dp["other"]:
        print("'Other' group for direct process emissions: ", groups_dp["other"])
    # From IPCC AR5
    GWP = {
        "CO2": 1,
        "CH4": 28,
        "N2O": 265,
    }

    def abatement_for_process(process_id):
        if process_id == "Incineration":
            return 1 - a_ccs_incineration
        elif process_id == "Mismanagement":
            # Never abated, by definition
            return 1
        elif process_id in PROCESS_GROUPS["biomass"]:
            # Sequestered (negative) emissions, don't abate these!
            return 1
        else:
            return 1 - a_ccs_process_emissions

    for group_id, group_process_ids in groups_dp.items():
        for ghg in GWP.keys():
            # Emissions for this group of processes and GHG specifically. Units:
            # emissions factors are in t_GHG/t_product. Our emissions are
            # reported in kgCO2e.  So, need to multiply by 1000.
            #
            abatement = {
                process_id: abatement_for_process(process_id)
                for process_id in group_process_ids
            }
            original_emissions = 1000 * process_factor_sum(
                model, values, f"DirProcEmis_{ghg}_", group_process_ids
            )
            abated_emissions = 1000 * process_factor_sum(
                model, values, f"DirProcEmis_{ghg}_", group_process_ids, extra_factors=abatement
            )
            results[f"Emissions_DirectProcess_{ghg}_{group_id}"] = abated_emissions
            results["CCS"] += original_emissions - abated_emissions

        # Sum up by total GWP
        results[f"Emissions_DirectProcess_GWP_{group_id}"] = sum(
            GWP[ghg] * results[f"Emissions_DirectProcess_{ghg}_{group_id}"]
            for ghg in GWP.keys()
        )

    # Feedstock emissions
    #
    # Units: feedstock emissions are reported in units of kgCO2e/tonne, and our
    # mass flows are measured in tonnes, so everything works out nicely here.
    for group, object_id, process_id in feedstock_emissions_params:
        group_key = f"Emissions_Feedstock_{group}"
        if group_key not in results:
            results[group_key] = sy.S.Zero

        if process_id is not None:
            # Emissions factor refers to supplying process scale
            #
            # Sanity check
            pid = model._lookup_process(process_id)
            assert object_id in model.processes[pid].produces

            feedstock_used = values[model.Y[pid]]
        else:
            # Emissions factor refers to quantity of object production deficit.
            # object_production_deficit() returns a structural placeholder that
            # is normally resolved when a step is compiled; since this isn't
            # part of a model step, resolve it explicitly against `values`.
            feedstock_used = model.structure.resolve_structural_symbols(
                model.object_production_deficit(object_id), values
            )
        emissions = feedstock_used * sy.Symbol(f"EF_Feedstock_{object_id}")

        results[f"FeedstockInput_{object_id}"] = feedstock_used
        results[f"FeedstockEmissionsByObject_{object_id}"] = emissions
        results[group_key] += emissions

    # Emissions by source
    def add_sources(source):
        return sum(
            emissions
            for k, emissions in results.items()
            if k.startswith(f"Emissions_{source}_")
        )

    results[f"EmissionsBySource_Elec"] = add_sources("ElecReq")
    results[f"EmissionsBySource_NG"] = add_sources("NGReq")  # combustion and WTT
    results[f"EmissionsBySource_NGCombustion"] = add_sources("NGCombustion")
    results[f"EmissionsBySource_NGWTT"] = add_sources("NGWTT")
    results[f"EmissionsBySource_Feedstock"] = add_sources("Feedstock")
    results[f"EmissionsBySource_Direct"] = add_sources("DirectProcess_GWP")

    # Emissions by lifecycle stage
    def add_stages(stage):
        return sum(
            results[k]
            for k in [
                f"Emissions_NGReq_{stage}",  # combustion and WTT
                f"Emissions_ElecReq_{stage}",
                f"Emissions_DirectProcess_GWP_{stage}",
            ]
        )

    # Just the explicitly modelled part (not fertilisers) not including
    # end-of-life
    results["EmissionsByStage_feedstocks"] = results["EmissionsBySource_Feedstock"]
    results["EmissionsByStage_hydrogen"] = add_stages("green_hydrogen") + add_stages(
        "other_hydrogen"
    )
    results["EmissionsByStage_primary_production"] = add_stages("primary_production")
    results["EmissionsByStage_biomass"] = add_stages("biomass")
    results["EmissionsByStage_organic_synthesis"] = add_stages("organic_synthesis")
    results["EmissionsByStage_downstream"] = add_stages("downstream")
    results["EmissionsByStage_end_of_life"] = add_stages("end_of_life")

    # For summary stat -- three big groups
    results["GHG_biomass_nonfertiliser"] = results["EmissionsByStage_biomass"]
    results["GHG_production_nonfertiliser"] = sum(
        results[f"EmissionsByStage_{k}"]
        for k in [
            "feedstocks",
            "hydrogen",
            "primary_production",
            "organic_synthesis",
            "downstream",
        ]
    )
    results["GHG_use_nonfertiliser"] = sy.S.Zero
    results["GHG_eol_nonfertiliser"] = results["EmissionsByStage_end_of_life"]
    results["GHG_nonfertiliser"] = (
        results["GHG_production_nonfertiliser"] + results["GHG_eol_nonfertiliser"]
    )

    return results


def define_polymer_model_production(model, polymer_demand=None):
    """Define the production part of the polymer model.

    This starts from demand for polymers, and works upstream. This part can be
    used in isolation from the main model (including use-phase and
    end-of-life) for testing.
    """
    # Balance polymer production now that end-of-life recycling is known -- but only
    # as far as primary chemicals, which are more complicated and dealt with below.
    for i, object_id in enumerate(polymer_objects):
        polymer_from_primary_chemicals(
            model,
            object_id,
            # Use deficit in model unless explicitly specified
            amount=polymer_demand[i] if polymer_demand is not None else None,
        )

    # Add in extra demand for chemicals which have not been explicitly included
    # in the demand for polymers above
    add_other_consumption(model)

    # Balance demand for intermediate chemicals (HydrogenCyanide)
    balance_intermediate_chemicals_from_primary_chemicals(model)

    # We already have some production of primary chemicals (e.g. ethylene) from
    # chemical recycling; now we need to decide how the remaining required
    # quantity of ethylene, propylene, BTX, etc should be made.
    #
    # Start with ethylene, since steam cracking produces many of the other
    # primary chemicals at the same time, then deploy other processes as needed
    # to balance production of the other primary chemicals.

    # Ethylene production: in preference order, produce from biomass (up to
    # capacity limit), from methyl alcohol (up to capacity limit), and then the
    # rest from ethane/naphtha.
    ethylene_from_biomass(model)
    ethylene_from_methyl_alcohol(model)
    ethylene_from_paraffins(model)

    # Now for BTX. After using up any spare pyrolysis gasoline via distillation,
    # use MTA (Methyl alcohol To Aromatics, i.e. BTX) up to capacity limit, then
    # if MTA has not produced enough xylenes yet, fall back to production from
    # fossil naphtha
    btx_from_spare_pyrolysis_gasoline(model)
    btx_from_methyl_alcohol(model)
    btx_from_spare_toluene(model)
    btx_from_naphtha(model)

    # Produce extra required propylene and butadiene, if we don't already have
    # enough as byproducts from the steps above.
    on_purpose_propylene(model)
    on_purpose_butadiene(model)

    ####### UPSTREAM PRODUCTION ####################

    # Produce methyl alcohol from hydrogen (if we have capacity) or else via
    # syngas. The required syngas is also produced here.
    methyl_alcohol_production(model)

    # Produce hydrogen needed for other uses (not for methyl alcohol -- already
    # done above). The reason this is done in two stages is that methyl alcohol
    # from hydrogen only make sense if green hydrogen is available (otherwise
    # might as well go direct from syngas). For other uses we also prefer green
    # hydrogen is there is more available, but we may need to fall back on
    # producing blue/grey hydrogen here.
    hydrogen_production(model)

    # Produce fossil paraffins as needed. "Naphtha" is supplied from chemical
    # recycling -- although perhaps not strictly the same thing as the output of
    # the pyrolysis process, it's a reasonable approximation to how the outputs
    # of pyrolysis could substitute other chemicals.
    fossil_paraffins_production(model)


def define_polymer_model(
    model,
    recipe_data,
    processes_with_elec_req=None,
    processes_with_direct_emissions=None,
):
    # Configure Use processes to hold stocks
    for p in model.processes:
        if p.id.startswith("UseOf"):
            p.has_stock = True

    # Demand and EOL flows of products
    stock_model_flows(model)

    # Recycling
    eol_polymer_treatment(model)

    # Production
    define_polymer_model_production(model)

    ######## Calculate utility requirements and emissions ##########

    if processes_with_elec_req is None:
        processes_with_elec_req = []
    if processes_with_direct_emissions is None:
        processes_with_direct_emissions = []

    # Compile the steps added so far to get the accumulated X[j]/Y[j] state,
    # needed to compute the summary/emissions expressions below.
    values = model.build()._values

    flow_results = calc_flow_summaries(model, values)
    utility_reqs = calc_utility_requirements(model, values, processes_with_elec_req)
    emissions = calc_emissions(model, values, utility_reqs, processes_with_direct_emissions)
    other_results = {**flow_results, **utility_reqs, **emissions}

    return other_results
