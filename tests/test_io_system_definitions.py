"""Tests for loading a model structure directly from system definitions."""

from pathlib import Path
from textwrap import dedent

import pytest
from rdflib import Graph, Namespace, URIRef

pytest.importorskip(
    "sphinx_probs_rdf.loader",
    reason="needs sphinx_probs_rdf with its `loader` extra (flowprog[definitions])",
)

from sphinx_probs_rdf.loader import parse_markdown  # noqa: E402

from flowprog.io.rdf import load_structure as load_structure_from_rdf  # noqa: E402
from flowprog.io.system_definitions import (  # noqa: E402
    load_structure,
    select_markets,
    select_processes,
    structure_from_parsed_system,
    unit_table,
)

DATA = Path(__file__).parent / "data"
QUANTITYKIND = Namespace("http://qudt.org/vocab/quantitykind/")

#: The units `metals_system.md` needs on top of the kg/m2/m3/- defaults.
METALS_UNITS = unit_table({"kWh": (1, "Energy")})


def parse(*texts, units=None):
    """Parse `texts` as system definitions.

    Each fragment is dedented separately before being joined, so that fragments
    indented differently in the tests below can still be combined.
    """
    return parse_markdown(
        "\n\n".join(dedent(text) for text in texts),
        units=units if units is not None else METALS_UNITS,
    )


def build(*texts, units=None, **kwargs):
    return structure_from_parsed_system(parse(*texts, units=units), **kwargs)


def recipes_by_name(structure, recipe_data):
    """`recipe_data` keyed by readable names rather than symbol indices."""
    out = {}
    for process in structure.processes:
        j = structure.lookup_process(process.id)
        for role, symbol in (("consumes", structure.U), ("produces", structure.S)):
            for obj in getattr(process, role):
                i = structure.lookup_object(obj)
                out[(process.id, role, obj)] = recipe_data[symbol[i, j]]
    return out


TWO_PROCESSES = """
    ```{system:process} Mining
    ---
    consumes: |
    produces: |
      Ore                                      = 1 kg
    ---
    ```

    ```{system:process} Smelting
    ---
    consumes: |
      Ore                                      = 2 kg
    produces: |
      Metal                                    = 1 kg
    ---
    ```

    ```{system:object} Ore
    ```

    ```{system:object} Metal
    ```
"""


#: One process, with `waste` and `share` left to fill in so that the same recipe can
#: be written as amounts or as shares.
SORTING = """
    ```{{system:process}} Sorting
    ---
    consumes: |
      Waste                                    = {waste}
    produces: |
      Plastic                                  = 40 {share}
      Paper                                    = 60 {share}
    ---
    ```

    ```{{system:object}} Waste
    ```

    ```{{system:object}} Plastic
    ```

    ```{{system:object}} Paper
    ```
"""


class TestLoadingStructure:
    def test_processes_and_objects_are_loaded(self):
        structure, _ = build(TWO_PROCESSES)
        assert [p.id for p in structure.processes] == ["Mining", "Smelting"]
        assert [o.id for o in structure.objects] == ["Metal", "Ore"]
        assert structure.producers_of("Ore") == ["Mining"]
        assert structure.consumers_of("Ore") == ["Smelting"]

    def test_recipe_data_is_loaded(self):
        structure, recipe_data = build(TWO_PROCESSES)
        assert recipes_by_name(structure, recipe_data) == {
            ("Mining", "produces", "Ore"): 1.0,
            ("Smelting", "consumes", "Ore"): 2.0,
            ("Smelting", "produces", "Metal"): 1.0,
        }

    def test_processes_and_objects_are_ordered_by_name(self):
        """So that the same definitions always give the same structure."""
        structure, _ = build(
            """
            ```{system:process} Zeta
            ---
            produces: |
              Beta                                     = 1 kg
              Aardvark                                 = 1 kg
            ---
            ```

            ```{system:process} Alpha
            ---
            consumes: |
              Beta                                     = 1 kg
            produces: |
              Aardvark                                 = 1 kg
            ---
            ```

            ```{system:object} Beta
            ```

            ```{system:object} Aardvark
            ```
            """
        )
        assert [p.id for p in structure.processes] == ["Alpha", "Zeta"]
        assert [o.id for o in structure.objects] == ["Aardvark", "Beta"]
        assert structure.processes[1].produces == ["Aardvark", "Beta"]

    def test_only_objects_used_by_the_model_are_included(self):
        unused = """
            ```{system:object} Unused
            ```
        """
        assert "Unused" in parse(TWO_PROCESSES, unused).objects
        structure, _ = build(TWO_PROCESSES, unused)
        assert [o.id for o in structure.objects] == ["Metal", "Ore"]

    def test_load_structure_accepts_a_single_path(self):
        from_one = load_structure(DATA / "metals_system.md", units=METALS_UNITS)
        from_list = load_structure([DATA / "metals_system.md"], units=METALS_UNITS)
        assert [p.id for p in from_one[0].processes] == [
            p.id for p in from_list[0].processes
        ]


class TestSelectingProcesses:
    def test_processes_without_a_recipe_are_left_out(self):
        """Parent processes only group their children; they aren't flows themselves."""
        system = parse(
            """
            ```{system:process} Extraction
            :consumes: Gas
            :produces: Ore
            :become_parent: true
            ```
            """,
            TWO_PROCESSES,
        )
        assert "Extraction" in system.processes
        assert select_processes(system) == ["Mining", "Smelting"]
        structure, _ = structure_from_parsed_system(system)
        assert [p.id for p in structure.processes] == ["Mining", "Smelting"]

    def test_processes_can_be_given_explicitly(self):
        structure, _ = build(TWO_PROCESSES, processes=["Mining"])
        assert [p.id for p in structure.processes] == ["Mining"]
        assert [o.id for o in structure.objects] == ["Ore"]

    def test_unknown_process_is_an_error(self):
        with pytest.raises(ValueError, match="unknown process: \\['Melting'\\]"):
            build(TWO_PROCESSES, processes=["Melting"])

    def test_selecting_a_process_without_a_full_recipe_is_an_error(self):
        with pytest.raises(ValueError, match="'Extraction' item 'Gas' has no amount"):
            build(
                """
                ```{system:process} Extraction
                :consumes: Gas
                :produces: Ore
                ```

                ```{system:object} Gas
                ```

                ```{system:object} Ore
                ```
                """,
                processes=["Extraction"],
            )


class TestSelectingMarkets:
    def test_objects_both_produced_and_consumed_get_markets(self):
        structure, _ = build(TWO_PROCESSES)
        assert {o.id: o.has_market for o in structure.objects} == {
            "Metal": False,  # only produced -- leaves the model boundary
            "Ore": True,
        }

    def test_select_markets_reports_the_same_objects(self):
        system = parse(TWO_PROCESSES)
        assert select_markets(system, select_processes(system)) == ["Ore"]

    def test_markets_follow_the_selected_processes(self):
        """Ore is only consumed once Smelting is left out, so needs no market."""
        structure, _ = build(TWO_PROCESSES, processes=["Mining"])
        assert [o.id for o in structure.objects if o.has_market] == []

    def test_markets_can_be_given_explicitly(self):
        structure, _ = build(TWO_PROCESSES, markets=["Metal"])
        assert {o.id: o.has_market for o in structure.objects} == {
            "Metal": True,
            "Ore": False,
        }

    def test_no_markets(self):
        structure, _ = build(TWO_PROCESSES, markets=[])
        assert not any(o.has_market for o in structure.objects)

    def test_market_for_an_object_outside_the_model_is_an_error(self):
        with pytest.raises(ValueError, match="unknown market object: \\['Slag'\\]"):
            build(TWO_PROCESSES, markets=["Slag"])


class TestMetrics:
    def test_metric_is_inferred_from_the_units_used(self):
        structure, _ = build(
            """
            ```{system:process} PowerStation
            ---
            produces: |
              Electricity                              = 1 kWh
            ---
            ```

            ```{system:object} Electricity
            ```
            """
        )
        assert structure.objects[0].metric == QUANTITYKIND.Energy

    def test_amounts_are_scaled_into_the_metric_base_unit(self):
        units = unit_table({"MWh": (1000, "Energy"), "kWh": (1, "Energy")})
        structure, recipe_data = build(
            """
            ```{system:process} PowerStation
            ---
            produces: |
              Electricity                              = 2 MWh
            ---
            ```

            ```{system:object} Electricity
            ```
            """,
            units=units,
        )
        assert list(recipe_data.values()) == [2000.0]

    def test_object_only_ever_a_share_takes_the_metric_of_its_process(self):
        """A bare share is a fraction of the side's basis, so it fixes the metric."""
        structure, _ = build(SORTING.format(waste="1 m3", share="%"))
        assert {o.id: o.metric for o in structure.objects} == {
            "Paper": QUANTITYKIND.Volume,
            "Plastic": QUANTITYKIND.Volume,
            "Waste": QUANTITYKIND.Volume,
        }

    def test_object_with_no_metric_anywhere_falls_back_to_mass(self, caplog):
        structure, _ = build(
            """
            ```{system:process} Sorting
            ---
            consumes: |
              Waste                                    = 100 %
            produces: |
              Paper                                    = 100 %
            ---
            ```

            ```{system:object} Waste
            ```

            ```{system:object} Paper
            ```
            """
        )
        assert all(o.metric == QUANTITYKIND.Mass for o in structure.objects)
        assert "No basis for object" in caplog.text

    def test_recipe_in_the_wrong_metric_is_an_error(self):
        with pytest.raises(ValueError, match="measured in .*Mass.*metric .*Volume"):
            build(
                """
                ```{system:process} Mining
                ---
                produces: |
                  Ore                                      = 1 kg
                ---
                ```

                ```{system:object} Ore
                :basis: Volume
                ```
                """
            )


class TestUnitTable:
    def test_defaults_are_included(self):
        assert unit_table({}).lookup("kg").basis == str(QUANTITYKIND.Mass)

    def test_bare_name_resolves_against_quantity_kinds(self):
        unit = unit_table({"kWh": (1, "Energy")}).lookup("kWh")
        assert (unit.scale, unit.basis) == (1.0, str(QUANTITYKIND.Energy))

    def test_scale_may_be_omitted(self):
        assert unit_table({"kWh": "Energy"}).lookup("kWh").scale == 1.0

    def test_bracketed_uri_is_used_as_is(self):
        table = unit_table({"pkm": "<http://example.org/PassengerKM>"})
        assert table.lookup("pkm").basis == "http://example.org/PassengerKM"

    def test_prefixed_name_resolves_against_prefixes(self):
        table = unit_table(
            {"pkm": "ex:PassengerKM"}, prefixes={"ex": "http://example.org/"}
        )
        assert table.lookup("pkm").basis == "http://example.org/PassengerKM"

    def test_unknown_prefix_is_an_error(self):
        with pytest.raises(ValueError, match="unknown prefix 'ex'"):
            unit_table({"pkm": "ex:PassengerKM"})

    def test_basis_prefix_can_be_changed(self):
        table = unit_table({"kg": "mass"}, basis_prefix="http://example.org/basis/")
        assert table.lookup("kg").basis == "http://example.org/basis/mass"


class TestShares:
    """A `%` recipe is a dimensionless recipe, when everything is in one metric."""

    def test_shares_are_scaled_to_fractions(self):
        structure, recipe_data = build(SORTING.format(waste="1 kg", share="%"))
        assert recipes_by_name(structure, recipe_data) == {
            ("Sorting", "consumes", "Waste"): 1.0,
            ("Sorting", "produces", "Paper"): 0.6,
            ("Sorting", "produces", "Plastic"): 0.4,
        }

    def test_shares_on_both_sides(self):
        structure, recipe_data = build(SORTING.format(waste="100 %", share="%"))
        assert recipes_by_name(structure, recipe_data) == {
            ("Sorting", "consumes", "Waste"): 1.0,
            ("Sorting", "produces", "Paper"): 0.6,
            ("Sorting", "produces", "Plastic"): 0.4,
        }

    def test_shares_match_the_same_recipe_written_as_amounts(self):
        as_shares = build(SORTING.format(waste="1 kg", share="%"))
        as_amounts = build(
            SORTING.format(waste="1 kg", share="%").replace("40 %", "0.4 kg")
            .replace("60 %", "0.6 kg")
        )
        assert recipes_by_name(*as_shares) == recipes_by_name(*as_amounts)

    def test_shares_are_rejected_when_the_process_mixes_metrics(self):
        """Relating a kg side to a m3 side needs a coefficient a share can't express."""
        in_volume = """
            ```{system:object} Plastic
            :basis: http://qudt.org/vocab/quantitykind/Volume
            ```
        """
        system = parse(SORTING.format(waste="1 kg", share="%"), in_volume)
        assert system.objects["Plastic"].basis.endswith("Volume")  # redefined above
        with pytest.raises(NotImplementedError, match="not measure everything"):
            structure_from_parsed_system(system)

    def test_named_basis_shares_are_rejected(self):
        with pytest.raises(NotImplementedError, match="share in a named basis"):
            build(SORTING.format(waste="1 kg", share="%mass"))


class TestUnsupportedDefinitions:
    def test_repeated_object_on_one_side_is_rejected(self):
        with pytest.raises(ValueError, match="lists 'Ore' more than once"):
            build(
                """
                ```{system:process} Mining
                ---
                produces: |
                  Ore                                      = 1 kg
                  Ore                                      = 2 kg
                ---
                ```

                ```{system:object} Ore
                ```
                """
            )


class TestAgreesWithRdfLoader:
    """The two loaders should give the same model for the same system.

    `metals_system.md` and `metals_system.ttl` describe the same system, the latter
    in the form a Sphinx build of the former produces.
    """

    @pytest.fixture
    def from_definitions(self):
        return load_structure(DATA / "metals_system.md", units=METALS_UNITS)

    @pytest.fixture
    def from_rdf(self):
        graph = Graph()
        graph.parse(DATA / "metals_system.ttl", format="ttl")
        return load_structure_from_rdf(
            graph, URIRef("http://example.org/model/Model")
        )

    def test_same_processes(self, from_definitions, from_rdf):
        assert from_definitions[0].processes == from_rdf[0].processes

    def test_same_objects(self, from_definitions, from_rdf):
        assert from_definitions[0].objects == from_rdf[0].objects

    def test_same_recipe_data(self, from_definitions, from_rdf):
        assert recipes_by_name(*from_definitions) == recipes_by_name(*from_rdf)

    def test_the_model_is_not_trivial(self, from_definitions):
        """Guard against the comparisons above passing on empty structures."""
        structure, recipe_data = from_definitions
        assert [p.id for p in structure.processes] == [
            "Mining",
            "PowerStation",
            "Smelting",
        ]
        assert [o.id for o in structure.objects if o.has_market] == [
            "Electricity",
            "Ore",
        ]
        assert len(recipe_data) == 6


def test_load_from_rdf_module_still_works():
    """The pre-`flowprog.io` import path is kept working."""
    from flowprog.load_from_rdf import query_model_from_endpoint

    assert query_model_from_endpoint is load_structure_from_rdf
