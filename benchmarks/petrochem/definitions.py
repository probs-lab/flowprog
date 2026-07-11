"""Object/process id lists used by model_polymers.py.

Extracted from the calculator's sankey_definitions.py, which mixes these in
with a lot of unrelated Sankey-diagram (floweaver) plotting code -- these are
the only entries model_polymers.py actually needs.
"""

polymerisation_processes = [
    "PolymerisationOfHDPE",
    "PolymerisationOfLDPE",
    "PolymerisationOfLLDPE",
    "PolymerisationOfPP",
    "PolymerisationOfPolystyrene",
    "PolymerisationOfPVC",
    "PolymerisationOfPET",
    "PolymerisationOfPUR",
    "PolymerisationOfStyreneButadiene",
    "PolymerisationOfFibrePPA",
    "PolymerisationOfOtherPolymers",
]

polymer_objects = [
    "HDPEPolyethylene",
    "LDPEPolyethylene",
    "LLDPE",
    "PPPolypropylene",
    "PSPolystyrene",
    "PVCPolyvinylChloride",
    "PETPolyethyleneTerephthalatePolyesters",
    "Polyurethane",
    "SyntheticRubbers",
    "FibrePPA",
    "OtherPolymers",
]

organic_chemicals_synthesis = [
    "HydrogenCyanideSynthesis",
    "AceticAcidSynthesis",
    "PropyleneOxideSynthesis",
    "EthyleneOxideSynthesis",
    "CyclohexaneSynthesis",
    "OtherOrganicChemicalsSynthesis",
    "StyreneSynthesis",
    "VinylChlorideSynthesis",
    "IsophthalicAcidSynthesis",
    "TolueneDiisocyanateSynthesis",
    "EthyleneGlycolSynthesis",
    "TerephthalicAcidSynthesis",
    "HexamethylenediamineSynthesisFromButadiene",
    "AdipicAcidSynthesis",
    "AcrylonitrileSynthesis",
    "PolyolsSynthesis",
]
