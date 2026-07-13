"""Load the petrochemicals model structure, recipe, and benchmark scenarios
from model_data.json.

model_data.json was extracted once from the C-THRU Global Petrochemicals
Calculator (which normally builds this structure from RDF via
flowprog.load_from_rdf) -- see README.md.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import sympy as sy
from rdflib import URIRef

from flowprog import Process, Object, ElementaryExchange, ModelBuilder, ModelStructure
from flowprog.boundary_processes import Source, add_boundary_processes

DATA_PATH = Path(__file__).parent / "model_data.json"
MASS = URIRef("http://qudt.org/vocab/quantitykind/Mass")
ENERGY = URIRef("http://qudt.org/vocab/quantitykind/Energy")

# Elementary exchanges used by the B-based part of the emissions calculation
# in model_polymers.py:
# - Speciated direct process emissions (CO2/CH4/N2O, GWP-characterisable).
# - GHG_upstream_Feedstock: one pre-characterised aggregate for the
#   cradle-to-gate burden of every feedstock crossing the model boundary
#   (fossil paraffins via their existing OilRefining* processes, plus every
#   other feedstock -- NaturalGas, Coal, biomass, CapturedCarbonDioxide --
#   via a generated Source boundary process; see
#   FEEDSTOCK_BOUNDARY_OBJECTS/add_boundary_processes() below).
# - GHG_upstream_Electricity: cradle-to-gate carbon intensity of electricity
#   supply. Electricity and LowCarbonElectricity (a distinct quality feeding
#   WaterElectrolysisForHydrogen, the green-hydrogen route -- see plan section
#   3.3's own worked example) are distinct objects with distinct Source
#   processes, so each carries its own B value on this one exchange.
# - GHG_ProcessHeat_Combustion / GHG_upstream_ProcessHeat: process heat
#   (from natural gas combustion) is modelled as its own boundary-supplied
#   object rather than mixed into NaturalGas-the-feedstock's technosphere
#   flow, so that combustion (abatable via CCS) and well-to-tank upstream
#   burdens stay two distinct, separately-reportable exchanges. See
#   model_polymers.py's calc_emissions() for how these are read.
# - CO2_captured: the CCS counterpart of the abated exchanges above --
#   implementation plan section 4's pattern, applied directly rather than via
#   a technosphere CapturedCO2 object (no transport/storage process is
#   modelled). Every process with an abated B entry (direct process
#   emissions here; process heat combustion below) also gets a captured B
#   entry for the same raw quantity times the complementary (captured)
#   fraction, so "total CCS" is just an aggregate over this one exchange --
#   see model_polymers.py's calc_emissions().
FEEDSTOCK_EXCHANGE_ID = "GHG_upstream_Feedstock"
ELECTRICITY_EXCHANGE_ID = "GHG_upstream_Electricity"
PROCESS_HEAT_COMBUSTION_EXCHANGE_ID = "GHG_ProcessHeat_Combustion"
PROCESS_HEAT_WTT_EXCHANGE_ID = "GHG_upstream_ProcessHeat"
CAPTURED_CO2_EXCHANGE_ID = "CO2_captured"

ELEMENTARY_EXCHANGES = [
    ElementaryExchange(id="CO2", metric=MASS),
    ElementaryExchange(id="CH4", metric=MASS),
    ElementaryExchange(id="N2O", metric=MASS),
    ElementaryExchange(id=FEEDSTOCK_EXCHANGE_ID, metric=MASS),
    ElementaryExchange(id=ELECTRICITY_EXCHANGE_ID, metric=MASS),
    ElementaryExchange(id=PROCESS_HEAT_COMBUSTION_EXCHANGE_ID, metric=MASS),
    ElementaryExchange(id=PROCESS_HEAT_WTT_EXCHANGE_ID, metric=MASS),
    ElementaryExchange(id=CAPTURED_CO2_EXCHANGE_ID, metric=MASS),
]

@dataclass(frozen=True)
class Feedstock:
    """One row of the feedstock table: (object, emissions-reporting group,
    supplying process) -- the single source of truth for every feedstock
    crossing the model boundary, consumed both here (to generate the Source
    boundary process for objects with no pre-existing supplier, below) and by
    model_polymers.py's feedstock_emissions_params/FEEDSTOCK_PROCESS_OBJECTS
    (to generate the corresponding reporting groupings).

    :param group: Emissions-reporting group ("fossil"/"biomass"/"co2"), used
        by model_polymers.calc_emissions() for Emissions_Feedstock_{group}.
    :param boundary: True (default) if the object has no pre-existing
        supplying process in model_data.json, so build_structure() below
        generates one (a SourceOf{object_id} process, see boundary_specs).
        False for the 4 paraffins, which already have an explicit
        OilRefining{object_id} process -- model_polymers.py's
        add_direct_emission_and_feedstock_exchanges() adds the B burden to
        that existing process instead of generating a new one.
    """

    object_id: str
    group: str
    boundary: bool = True

    @property
    def process_id(self) -> str:
        return (
            f"SourceOf{self.object_id}"
            if self.boundary
            else f"OilRefining{self.object_id}"
        )


FEEDSTOCKS = [
    Feedstock("NaturalGas", "fossil"),
    Feedstock("Coal", "fossil"),
    Feedstock("Naphtha", "fossil", boundary=False),
    Feedstock("Ethane", "fossil", boundary=False),
    Feedstock("Propane", "fossil", boundary=False),
    Feedstock("Butane", "fossil", boundary=False),
    Feedstock("SugarCane", "biomass"),
    Feedstock("Maize", "biomass"),
    Feedstock("CornStover", "biomass"),
    Feedstock("SugarCaneBagasse", "biomass"),
    Feedstock("WheatStraw", "biomass"),
    Feedstock("RiceStraw", "biomass"),
    Feedstock("CapturedCarbonDioxide", "co2"),
]

# Feedstocks that cross the model boundary with no explicit supplying process
# already in model_data.json -- each gets a generated Source boundary process
# (see build_structure()). Naphtha/Ethane/Propane/Butane are deliberately
# excluded: they already have an explicit OilRefining* supplying process in
# the model (migrated in an earlier step -- see
# model_polymers.add_direct_emission_and_feedstock_exchanges()).
FEEDSTOCK_BOUNDARY_OBJECTS = [f.object_id for f in FEEDSTOCKS if f.boundary]

# Process consuming "low carbon" electricity (its own object/Source, distinct
# from ordinary grid Electricity) rather than ordinary Electricity -- the
# green-hydrogen route. All other processes in `processes_with_elec_req`
# consume ordinary Electricity. See ELECTRICITY_EXCHANGE_ID's docstring above.
LOW_CARBON_ELECTRICITY_PROCESSES = {"WaterElectrolysisForHydrogen"}

# Not a scenario parameter: fixed per Meys et al's minimal wind-based "green"
# electricity assumption (matches the value used by the pre-migration ad-hoc
# calc_emissions()).
LOW_CARBON_ELECTRICITY_EF = sy.Float(0.007)

# Natural-gas-fired process heat's combustion emissions are abated by the
# same CCS deployment rate as other utility combustion. Redefined locally
# (rather than imported from model_polymers) to keep structure.py's only
# dependency on model logic to this one symbol name.
a_ccs_utility_combustion = sy.Symbol("a_ccs_utility_combustion", nonnegative=True)

# Gross CV of natural gas (DEFRA data): 50.4 GJ/t = 50400 MJ/t. Converts the
# tonne-basis feedstock upstream EF into the MJ basis ProcessHeat is metered in.
NATURAL_GAS_ENERGY_CONTENT_MJ_PER_T = 50400


class _no_boundary_market_warning:
    """Suppress flowprog.boundary_processes' "no market" warning for the
    duration of one add_boundary_processes() call -- see the comment at its
    call site in build_structure() for why it's a deliberate false positive
    here."""

    def __enter__(self):
        self._logger = logging.getLogger("flowprog.boundary_processes")
        self._previous_level = self._logger.level
        self._logger.setLevel(logging.ERROR)

    def __exit__(self, *exc_info):
        self._logger.setLevel(self._previous_level)


def load_data() -> dict:
    with open(DATA_PATH) as f:
        return json.load(f)


def _symbolize_fragment(builder: ModelBuilder, fragment: dict) -> dict:
    """Convert an ID-based recipe fragment (as returned by
    add_boundary_processes) into the symbol-keyed format used elsewhere in
    this module's recipe_data."""
    symbols = {}
    for process_id, recipe in fragment.items():
        j = builder._lookup_process(process_id)
        for object_id, value in recipe.get("produces", {}).items():
            i = builder._lookup_object(object_id)
            symbols[builder.S[i, j]] = value
        for object_id, value in recipe.get("consumes", {}).items():
            i = builder._lookup_object(object_id)
            symbols[builder.U[i, j]] = value
        for exchange_id, value in recipe.get("exchanges", {}).items():
            e = builder._lookup_exchange(exchange_id)
            symbols[builder.B[e, j]] = value
    return symbols


def build_structure(data: dict) -> tuple[ModelBuilder, dict]:
    """Return (ModelBuilder, recipe_data) for the model in `data`."""
    struct = data["structure"]
    processes_with_elec_req = set(data["processes_with_elec_req"])

    processes = []
    for p in struct["processes"]:
        consumes = list(p["consumes"])
        if p["id"] in processes_with_elec_req:
            if p["id"] in LOW_CARBON_ELECTRICITY_PROCESSES:
                consumes.append("LowCarbonElectricity")
            else:
                consumes.append("Electricity")
            consumes.append("ProcessHeat")
        processes.append(Process(id=p["id"], produces=p["produces"], consumes=consumes))

    objects = [
        Object(id=o["id"], metric=MASS, has_market=o["has_market"])
        for o in struct["objects"]
    ] + [
        Object(id="Electricity", metric=ENERGY, has_market=True),
        Object(id="LowCarbonElectricity", metric=ENERGY, has_market=True),
        Object(id="ProcessHeat", metric=ENERGY, has_market=True),
    ]

    base_structure = ModelStructure(processes, objects, ELEMENTARY_EXCHANGES)

    boundary_specs = [
        Source(
            f.object_id,
            exchanges={
                FEEDSTOCK_EXCHANGE_ID: sy.Symbol(f"EF_Feedstock_{f.object_id}")
            },
        )
        for f in FEEDSTOCKS
        if f.boundary
    ] + [
        Source(
            "Electricity",
            exchanges={ELECTRICITY_EXCHANGE_ID: sy.Symbol("EF_Utility_Electricity")},
        ),
        Source(
            "LowCarbonElectricity",
            exchanges={ELECTRICITY_EXCHANGE_ID: LOW_CARBON_ELECTRICITY_EF},
        ),
        Source(
            "ProcessHeat",
            exchanges={
                PROCESS_HEAT_COMBUSTION_EXCHANGE_ID: (
                    sy.Symbol("EF_Utility_NaturalGas") * (1 - a_ccs_utility_combustion)
                ),
                PROCESS_HEAT_WTT_EXCHANGE_ID: (
                    sy.Symbol("EF_Feedstock_NaturalGas")
                    / NATURAL_GAS_ENERGY_CONTENT_MJ_PER_T
                ),
                CAPTURED_CO2_EXCHANGE_ID: (
                    sy.Symbol("EF_Utility_NaturalGas") * a_ccs_utility_combustion
                ),
            },
        ),
    ]
    # The 9 feedstock objects deliberately keep has_market=False: nothing
    # else in model_polymers.py recurses backward into them automatically
    # (every existing pull_production() call that reaches them already stops
    # there, via until_objects or -- for Coal/biomass/CapturedCarbonDioxide,
    # which had no producer at all pre-migration -- has_market=False itself).
    # Their market is instead closed explicitly by
    # dispatch_boundary_processes() pulling the remaining production deficit
    # through each Source process (canonical pattern, plan section 3.2).
    # Flipping has_market=True would risk those existing calls silently
    # auto-recursing into the new Source processes mid-step instead, which
    # is unnecessary and harder to reason about than one explicit dispatch
    # step per object -- so the "almost certainly a mistake" warning below is
    # a deliberate, reviewed false positive for this specific case.
    with _no_boundary_market_warning():
        new_structure, fragment = add_boundary_processes(base_structure, boundary_specs)

    builder = ModelBuilder.from_structure(new_structure)

    recipe_data = {}
    for entry in struct["recipe"]:
        i = builder.structure.lookup_object(entry["object"])
        j = builder.structure.lookup_process(entry["process"])
        base = builder.S if entry["role"] == "produces" else builder.U
        recipe_data[base[i, j]] = entry["value"]

    recipe_data.update(_symbolize_fragment(builder, fragment))

    for process_id in processes_with_elec_req:
        j = builder._lookup_process(process_id)
        elec_object_id = (
            "LowCarbonElectricity"
            if process_id in LOW_CARBON_ELECTRICITY_PROCESSES
            else "Electricity"
        )
        i_elec = builder._lookup_object(elec_object_id)
        recipe_data[builder.U[i_elec, j]] = sy.Symbol(f"ElecReq_{process_id}")
        i_heat = builder._lookup_object("ProcessHeat")
        recipe_data[builder.U[i_heat, j]] = sy.Symbol(f"NGReq_{process_id}")

    return builder, recipe_data
