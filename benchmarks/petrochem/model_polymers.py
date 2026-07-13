import sympy as sy
from flowprog import merge_activities
from flowprog.allocation import PassThrough
from flowprog.reporting import Grouping, Report

from utils import (
    def_scalar_param,
    def_vector_param,
    pull_production_with_capacity_limit,
)
from structure import (
    FEEDSTOCK_EXCHANGE_ID,
    FEEDSTOCK_BOUNDARY_OBJECTS,
    FEEDSTOCKS,
    ELECTRICITY_EXCHANGE_ID,
    PROCESS_HEAT_COMBUSTION_EXCHANGE_ID,
    PROCESS_HEAT_WTT_EXCHANGE_ID,
    CAPTURED_CO2_EXCHANGE_ID,
    a_ccs_utility_combustion,
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
# a_ccs_utility_combustion is imported from structure.py above: it's baked
# into SourceOfProcessHeat's combustion B value there (see build_structure()),
# and re-exposed here purely so this stays a complete list of the model's
# scenario parameters.
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
paraffins = [f.object_id for f in FEEDSTOCKS if not f.boundary]

# Parameters for calculating feedstock emissions -- derived from structure.py's
# FEEDSTOCKS table, the single source of truth for which object needs which
# supplying process (also used there to generate each boundary object's
# Source process).
feedstock_emissions_params = [(f.group, f.object_id, f.process_id) for f in FEEDSTOCKS]

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


# Every object supplied by a generated boundary process (see structure.py's
# build_structure()): the 9 feedstocks with no pre-existing supplying process,
# plus the 3 utility objects (Electricity/LowCarbonElectricity/ProcessHeat).
BOUNDARY_SUPPLIED_OBJECTS = FEEDSTOCK_BOUNDARY_OBJECTS + [
    "Electricity",
    "LowCarbonElectricity",
    "ProcessHeat",
]

# The utility Source processes are *conduits* in the emissions contribution
# analysis: their burdens are reattributed to the consuming processes via
# PassThrough (see calc_emissions()). The feedstock Sources are deliberately
# NOT in this list -- feedstock burdens are reported producer-side, as their
# own "feedstocks" category.
UTILITY_SOURCE_PROCESSES = [
    "SourceOfElectricity",
    "SourceOfLowCarbonElectricity",
    "SourceOfProcessHeat",
]


def dispatch_boundary_processes(model):
    """Dispatch: pull the remaining production deficit of every
    boundary-supplied object through its Source process (canonical pattern,
    flowprog implementation plan section 3.2). Must run after every step that
    consumes these objects, so the deficit reflects total demand and the
    market fully balances.
    """
    for object_id in BOUNDARY_SUPPLIED_OBJECTS:
        model.add(
            model.pull_process_output(
                f"SourceOf{object_id}",
                object_id,
                model.object_production_deficit(object_id),
            ),
            label=f"Boundary supply of {object_id}",
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


# From IPCC AR5
GWP = {
    "CO2": 1,
    "CH4": 28,
    "N2O": 265,
}

# Lifecycle stages: the PROCESS_GROUPS partition plus an "other" catch-all for
# every process not in a named group (boundary Sources, Use processes, ...).
STAGES = list(PROCESS_GROUPS) + ["other"]

# Polymers with a mechanical-recycling process (the rest are only ever mixed /
# chemically recycled) -- their recycled output is reported separately from
# virgin polymerisation. See calc_flow_summaries().
RECYCLED_POLYMERS = {
    "LDPEPolyethylene",
    "LLDPE",
    "HDPEPolyethylene",
    "PPPolypropylene",
    "PSPolystyrene",
    "PVCPolyvinylChloride",
    "PETPolyethyleneTerephthalatePolyesters",
    "FibrePPA",
}

# Emissions "source" categories, as a grouping over elementary-exchange ids.
# CO2_captured is placed in "other" -- it is a CCS diagnostic, not a GWP source
# (see calc_emissions()'s CCS total), and GWP_ALL characterises it to zero.
EMISSIONS_SOURCE_GROUPING = {
    "Elec": {ELECTRICITY_EXCHANGE_ID},
    "NGCombustion": {PROCESS_HEAT_COMBUSTION_EXCHANGE_ID},
    "NGWTT": {PROCESS_HEAT_WTT_EXCHANGE_ID},
    "Feedstock": {FEEDSTOCK_EXCHANGE_ID},
    "Direct": set(GWP),  # direct process emissions CO2/CH4/N2O
    "other": {CAPTURED_CO2_EXCHANGE_ID},
}

# One combined characterisation to kgCO2e: GWP factors for the speciated
# direct-emission gases; 1 for the already-kgCO2e upstream / combustion
# exchanges; CO2_captured omitted (-> 0, it is a mass diagnostic, not a
# warming contribution).
GWP_ALL = {
    **GWP,
    ELECTRICITY_EXCHANGE_ID: 1,
    PROCESS_HEAT_COMBUSTION_EXCHANGE_ID: 1,
    PROCESS_HEAT_WTT_EXCHANGE_ID: 1,
    FEEDSTOCK_EXCHANGE_ID: 1,
}

# Feedstock supplying processes grouped by emissions-reporting group, from the
# FEEDSTOCKS table (structure.py).
FEEDSTOCK_GROUPS = list(dict.fromkeys(f.group for f in FEEDSTOCKS))
FEEDSTOCK_GROUPING = {
    group: {f.process_id for f in FEEDSTOCKS if f.group == group}
    for group in FEEDSTOCK_GROUPS
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


# FEEDSTOCKS entries with an explicit supplying process that pre-dates the
# boundary-process migration (Naphtha/Ethane/Propane/Butane via OilRefining*)
# -- these carry an *added* B burden on an otherwise-ordinary existing
# process (see add_direct_emission_and_feedstock_exchanges()). Everything
# else (NaturalGas, Coal, biomass, CapturedCarbonDioxide) is supplied by a
# generated Source boundary process instead, whose B value is set directly as
# part of its spec in structure.py's build_structure() --
# add_direct_emission_and_feedstock_exchanges() does not need to (and must
# not, to avoid setting the same recipe entry twice) touch those.
FEEDSTOCK_PROCESS_OBJECTS = {
    f.process_id: f.object_id for f in FEEDSTOCKS if not f.boundary
}


def add_direct_emission_and_feedstock_exchanges(
    model, recipe_data, processes_with_direct_emissions
):
    """Populate B (elementary exchange) recipe entries for the parts of the
    ad-hoc emissions calculation that map cleanly onto a per-process burden
    without any structural change to the model:

    - Direct process emissions (CO2, CH4, N2O), abated per abatement_for_process(),
      with the captured (1 - abatement_for_process()) counterpart recorded
      against CAPTURED_CO2_EXCHANGE_ID so "total CCS" is an aggregate over it
      (implementation plan section 4's pattern) rather than a parallel
      unabated calculation.
    - Feedstock emissions for objects with an explicit supplying process
      already in the model (Naphtha/Ethane/Propane/Butane via OilRefining*).

    Mutates `recipe_data` in place, and declares the exchanges on the affected
    processes. Must be called before `model.build(recipe_data)`.

    """

    def declare(process, exchange_ids):
        process.exchanges.extend(
            e for e in exchange_ids if e not in process.exchanges
        )

    e_captured = model.structure.lookup_exchange(CAPTURED_CO2_EXCHANGE_ID)
    for process_id in processes_with_direct_emissions:
        j = model._lookup_process(process_id)
        declare(model.processes[j], [*GWP, CAPTURED_CO2_EXCHANGE_ID])
        abatement = abatement_for_process(process_id)
        captured = 1 - abatement
        # Units: DirProcEmis_* is t_GHG/t_product; results are reported
        # in kgCO2e, hence the x1000.
        for ghg in GWP:
            e = model.structure.lookup_exchange(ghg)
            recipe_data[model.B[e, j]] = (
                1000 * sy.Symbol(f"DirProcEmis_{ghg}_{process_id}") * abatement
            )
        # CO2_captured sums the CO2/CH4/N2O *masses* (t) unweighted into one
        # "captured" quantity -- physically odd (it books captured N2O at mass
        # parity with CO2), but this faithfully reproduces the legacy CCS
        # definition, which was likewise GWP-unweighted. Inherited behaviour.
        recipe_data[model.B[e_captured, j]] = 1000 * captured * sum(
            sy.Symbol(f"DirProcEmis_{ghg}_{process_id}") for ghg in GWP
        )

    e_feedstock = model.structure.lookup_exchange(FEEDSTOCK_EXCHANGE_ID)
    for process_id, object_id in FEEDSTOCK_PROCESS_OBJECTS.items():
        j = model._lookup_process(process_id)
        assert object_id in model.processes[j].produces
        declare(model.processes[j], [FEEDSTOCK_EXCHANGE_ID])
        recipe_data[model.B[e_feedstock, j]] = sy.Symbol(f"EF_Feedstock_{object_id}")


def calc_flow_summaries(structure):
    """Calculate summary metrics based on mass flows."""

    results = {}

    # Total polymer production, split by route: virgin (PolymerisationOf*) vs
    # recycled (MechanicalRecyclingOf*AtEOL). Each is a Report.production()
    # total filtered to the one producing process, mirroring the feedstock
    # reads.
    results["Production_polymers_virgin"] = sum(
        Report.production(structure, polymer).filter(process=process).total()
        for polymer, process in zip(polymer_objects, polymerisation_processes)
    )
    results["Production_polymers_recycled"] = sum(
        Report.production(structure, polymer)
        .filter(process=f"MechanicalRecyclingOf{polymer}AtEOL")
        .total()
        for polymer in polymer_objects
        if polymer in RECYCLED_POLYMERS
    )

    # Process capacity for every process: its Max(input, output) activity.
    for j, p in enumerate(structure.processes):
        results[f"ProcessThroughput_{p.id}"] = sy.Max(structure.X[j], structure.Y[j])

    return results


def calc_utility_requirements(structure, stage_grouping):
    """Utility (electricity/process-heat) requirements by lifecycle stage.

    The technosphere analogue of the elementary-flow aggregation:
    ElecReq_{stage}/NGReq_{stage} are a group-by over Electricity/
    LowCarbonElectricity/ProcessHeat consumption, reading straight off the
    U[i,j] entries structure.py already populated (the ElecReq_{process_id}/
    NGReq_{process_id} symbols) via Report.consumption() instead of
    re-deriving those symbol names by hand.
    """

    def consumption_by_stage(object_id):
        return (
            Report.consumption(structure, object_id)
            .with_group("stage", stage_grouping, on="process")
            .by("stage")
        )

    elec = consumption_by_stage("Electricity")
    low_carbon_elec = consumption_by_stage("LowCarbonElectricity")
    heat = consumption_by_stage("ProcessHeat")

    other_results = {}
    for stage in STAGES:
        other_results[f"ElecReq_{stage}"] = elec.get(
            stage, sy.S.Zero
        ) + low_carbon_elec.get(stage, sy.S.Zero)
        other_results[f"NGReq_{stage}"] = heat.get(stage, sy.S.Zero)

    other_results["ElecReq"] = sum(
        other_results[f"ElecReq_{stage}"] for stage in STAGES
    )
    other_results["NGReq"] = sum(other_results[f"NGReq_{stage}"] for stage in STAGES)

    return other_results


def calc_emissions(structure, stage_grouping):
    """All emissions results, as views over `flows` -- the grouped Report on
    the (utility-reattributed) elementary-flow table.

    Two stage pivots do most of the work. Both run over the same reattributed
    flows: `flows` wraps PassThrough's table, which moves each utility
    Source's burden (Electricity/LowCarbonElectricity/ProcessHeat) onto its
    consuming processes -- so it lands in the consumer's stage, not the
    Source's -- while direct-emission and feedstock burdens sit producer-side
    and stay put. Green hydrogen needs no special case: it consumes
    LowCarbonElectricity (its own object/Source), so its low-carbon EF is
    routed structurally.

    - `by_stage_exchange` (uncharacterised -- fine, since "exchange" is one
      of the group keys): the per-gas direct emissions and the CO2_captured
      diagnostic, which GWP_ALL deliberately drops.
    - `by_stage_source` (GWP_ALL-characterised, exchanges collapsed to source
      categories): every kgCO2e source figure, with CO2/CH4/N2O rolled into
      one "Direct" column. EmissionsBySource_* are its column sums and
      EmissionsByStage_* its row sums -- no prefix-scanning a results dict.
    """

    results = {}

    process_ids = [p.id for p in structure.processes]
    feedstock_grouping = Grouping.complete(FEEDSTOCK_GROUPING, process_ids)

    pass_through = PassThrough(structure, UTILITY_SOURCE_PROCESSES)
    flows = (
        Report.elementary_flows(structure, pass_through.elementary_flows())
        .with_group("stage", stage_grouping, on="process")
        .with_group("feedstock_group", feedstock_grouping, on="process")
        .with_group("source", EMISSIONS_SOURCE_GROUPING, on="exchange")
    )

    by_stage_exchange = flows.by("stage", "exchange").to_dict()
    by_stage_source = (
        flows.characterise(GWP_ALL, name="GWPall").by("stage", "source").to_dict()
    )

    def stage_exchange(stage, exchange_id):
        return by_stage_exchange.get((stage, exchange_id), sy.S.Zero)

    def stage_source(stage, source):
        return by_stage_source.get((stage, source), sy.S.Zero)

    # Utility emissions by stage -- the source pivot's Elec/NGCombustion/NGWTT
    # columns (GWP_ALL factor 1, i.e. the already-kgCO2e burden). Process heat
    # keeps combustion (CCS-abated) and well-to-tank as two distinct sources;
    # captured CO2 feeds the CCS total below, not a per-stage figure here.
    for stage in STAGES:
        results[f"Emissions_ElecReq_{stage}"] = stage_source(stage, "Elec")
        combustion = stage_source(stage, "NGCombustion")
        wtt = stage_source(stage, "NGWTT")
        results[f"Emissions_NGCombustion_{stage}"] = combustion
        results[f"Emissions_NGWTT_{stage}"] = wtt
        results[f"Emissions_NGReq_{stage}"] = combustion + wtt

    # Direct process emissions by stage: per-gas (unweighted B*Y from the
    # exchange pivot) plus the GWP-weighted roll-up (the source pivot's
    # "Direct" column). B-sparsity intersects with the processes that actually
    # carry direct-emission entries, so no explicit process-group
    # intersection is needed.
    for stage in STAGES:
        for ghg in GWP:
            results[f"Emissions_DirectProcess_{ghg}_{stage}"] = stage_exchange(
                stage, ghg
            )
        results[f"Emissions_DirectProcess_GWP_{stage}"] = stage_source(stage, "Direct")

    # Stage aggregates (sum over stages).
    results["Emissions_ElecReq"] = sum(
        results[f"Emissions_ElecReq_{s}"] for s in STAGES
    )
    results["Emissions_NGCombustion"] = sum(
        results[f"Emissions_NGCombustion_{s}"] for s in STAGES
    )
    results["Emissions_NGWTT"] = sum(results[f"Emissions_NGWTT_{s}"] for s in STAGES)
    results["Emissions_NGReq"] = sum(results[f"Emissions_NGReq_{s}"] for s in STAGES)

    # Total CO2 captured by CCS -- the one exchange GWP_ALL omits. Direct
    # process emissions and process-heat combustion both carry a captured B
    # entry on CAPTURED_CO2_EXCHANGE_ID (implementation plan section 4), so
    # "total CCS" is just that column summed across stages.
    results["CCS"] = sum(
        stage_exchange(stage, CAPTURED_CO2_EXCHANGE_ID) for stage in STAGES
    )

    # Feedstock emissions.
    #
    # Units: feedstock EFs are kgCO2e/tonne and mass flows are tonnes. Feedstock
    # Sources are producer-side (not reattributed, unlike the utility Sources),
    # so these read straight off GHG_upstream_Feedstock:
    #  - per group, from the feedstock_group x exchange pivot;
    #  - per object, from the process x exchange pivot (one process each);
    #  - FeedstockInput from Report.production() -- the process filter is
    #    load-bearing: Naphtha is co-produced by chemical recycling, which is
    #    not fossil feedstock input. (production() == Y[j] here as every
    #    supplying S is 1.)
    feedstock_by_group = flows.by("feedstock_group", "exchange").to_dict()
    feedstock_by_process = flows.by("process", "exchange").to_dict()
    for group in FEEDSTOCK_GROUPS:
        results[f"Emissions_Feedstock_{group}"] = feedstock_by_group.get(
            (group, FEEDSTOCK_EXCHANGE_ID), sy.S.Zero
        )
    for f in FEEDSTOCKS:
        results[f"FeedstockInput_{f.object_id}"] = (
            Report.production(structure, f.object_id)
            .filter(process=f.process_id)
            .total()
        )
        results[f"FeedstockEmissionsByObject_{f.object_id}"] = feedstock_by_process.get(
            (f.process_id, FEEDSTOCK_EXCHANGE_ID), sy.S.Zero
        )

    # Emissions by source -- column sums of the (stage, source) pivot.
    def source_total(source):
        return sum(stage_source(stage, source) for stage in STAGES)

    results["EmissionsBySource_Elec"] = source_total("Elec")
    results["EmissionsBySource_NGCombustion"] = source_total("NGCombustion")
    results["EmissionsBySource_NGWTT"] = source_total("NGWTT")
    results["EmissionsBySource_NG"] = source_total("NGCombustion") + source_total(
        "NGWTT"
    )
    results["EmissionsBySource_Feedstock"] = source_total("Feedstock")
    results["EmissionsBySource_Direct"] = source_total("Direct")

    # Emissions by lifecycle stage -- row sums of the (stage, source) pivot
    # over the emitting sources. Feedstock is reported as its own pseudo-stage
    # (below), not folded into the per-stage rows.
    def stage_total(stage):
        return sum(
            stage_source(stage, source)
            for source in ("Elec", "NGCombustion", "NGWTT", "Direct")
        )

    results["EmissionsByStage_feedstocks"] = results["EmissionsBySource_Feedstock"]
    results["EmissionsByStage_hydrogen"] = stage_total("green_hydrogen") + stage_total(
        "other_hydrogen"
    )
    results["EmissionsByStage_primary_production"] = stage_total("primary_production")
    results["EmissionsByStage_biomass"] = stage_total("biomass")
    results["EmissionsByStage_organic_synthesis"] = stage_total("organic_synthesis")
    results["EmissionsByStage_downstream"] = stage_total("downstream")
    results["EmissionsByStage_end_of_life"] = stage_total("end_of_life")

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

    # Balance every remaining boundary-supplied object (feedstocks with no
    # pre-existing supplying process, plus the utility objects) against the
    # demand accumulated by all the production steps above.
    dispatch_boundary_processes(model)


def define_polymer_model(
    model,
    recipe_data,
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

    if processes_with_direct_emissions is None:
        processes_with_direct_emissions = []

    # Populate B (elementary exchange) recipe entries for direct process
    # emissions and explicit-process feedstocks (mutates recipe_data; must
    # happen before the final model.build(recipe_data) call in run_benchmark.py).
    add_direct_emission_and_feedstock_exchanges(
        model, recipe_data, processes_with_direct_emissions
    )

    # Compile the steps added so far to get the accumulated X[j]/Y[j] state,
    # needed to compute the summary/emissions expressions below. Passing
    # recipe_data here only *stores* it on the compiled model (set_recipe();
    # no substitution into the accumulated values happens at build time) --
    # it's needed so PassThrough/Report below can read recipe values. All
    # emissions expressions stay raw (intermediates unresolved), deferred to
    # the final model.lambdify(expressions=other_results) call, which
    # resolves everything in one efficient, CSE-aware pass.
    compiled = model.build(recipe_data)

    # One shared Report drives the emissions summaries below: the utility
    # Source processes' burdens are reattributed to their consumers via
    # PassThrough (table substitution -- see calc_emissions()'s docstring),
    # with three grouping columns:
    #  - "stage": the PROCESS_GROUPS partition + an explicit "other"
    #    catch-all over the remaining processes (boundary Sources, Use, ...);
    #  - "source": elementary exchanges grouped into emission-source categories;
    #  - "feedstock_group": feedstock supplying processes by reporting group.
    # Characterisation (GWP_ALL -> kgCO2e) is applied per view in
    # calc_emissions(), not baked in here.
    process_ids = [p.id for p in compiled.processes]
    stage_grouping = Grouping.complete(PROCESS_GROUPS, process_ids)

    flow_results = calc_flow_summaries(compiled.structure)
    utility_reqs = calc_utility_requirements(compiled.structure, stage_grouping)
    emissions = calc_emissions(compiled.structure, stage_grouping)
    other_results = {**flow_results, **utility_reqs, **emissions}

    return other_results
