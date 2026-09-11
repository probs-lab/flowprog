"""Tests for flowprog.allocation.contributions."""

from itertools import combinations

import numpy as np
import pytest
import sympy as sy
from rdflib import URIRef

from flowprog import ElementaryExchange, ModelBuilder, ModelStructure, Object, Process
from flowprog.allocation import (
    AllocatedSystem,
    ByValue,
    Excluding,
    ProcessSet,
    Scope,
)

MASS = URIRef("http://qudt.org/vocab/quantitykind/Mass")


def MObject(id, *args, **kwargs):
    return Object(id, MASS, *args, **kwargs)


# ============================================================================
# Toy model:
#
#   SourceOfOre         --Ore---------> MakeSteel --Steel--> MakeWidget --> Widget
#   SourceOfElectricity --Electricity->     |      --Slag-->     ^  |
#                       ---------------------------------------->  +-> Scrap
#                                    Water (nothing produces it) ->
#
# Per widget: Y_MakeWidget=1, Y_MakeSteel=2, Ore=3, Electricity=11.
# With Scrap excluded from bearing burden, MakeSteel's two outputs split its
# burden equally, so per unit of Steel or Slag:
#
#   mu_Electricity = 0.05,  mu_Ore = 0.3
#   burden leaving MakeSteel (per operation) = 1.2 + 1.5*0.3 + 4*0.05 = 1.85
#   mu_Steel = mu_Slag = 0.5 * 1.85 = 0.925
#   mu_Widget = 0.4 + 2*0.925 + 2*0.925 + 3*0.05 = 4.25
#
# Water has no producer, so it carries no burden and is reported as unsupplied.
# ============================================================================

PROCESSES = [
    Process(
        "MakeWidget",
        produces=["Widget", "Scrap"],
        consumes=["Steel", "Slag", "Electricity", "Water"],
        exchanges=["CO2"],
    ),
    Process(
        "MakeSteel",
        produces=["Steel", "Slag"],
        consumes=["Ore", "Electricity"],
        exchanges=["CO2"],
    ),
    Process(
        "SourceOfElectricity", produces=["Electricity"], consumes=[], exchanges=["CO2"]
    ),
    Process("SourceOfOre", produces=["Ore"], consumes=[], exchanges=["CO2"]),
]

RECIPE = {
    "MakeWidget": {
        "consumes": {"Steel": 2.0, "Slag": 2.0, "Electricity": 3.0, "Water": 0.5},
        "produces": {"Widget": 1.0, "Scrap": 0.25},
        "exchanges": {"CO2": 0.4},
    },
    "MakeSteel": {
        "consumes": {"Ore": 1.5, "Electricity": 4.0},
        "produces": {"Steel": 1.0, "Slag": 1.0},
        "exchanges": {"CO2": 1.2},
    },
    "SourceOfElectricity": {
        "produces": {"Electricity": 1.0},
        "exchanges": {"CO2": 0.05},
    },
    "SourceOfOre": {"produces": {"Ore": 1.0}, "exchanges": {"CO2": 0.3}},
}

RULE = Excluding({"Scrap"}, ByValue())


def build_toy_model():
    objects = [
        MObject("Widget", has_market=True),
        MObject("Scrap"),
        MObject("Steel", has_market=True),
        MObject("Slag"),
        MObject("Electricity", has_market=True),
        MObject("Ore", has_market=True),
        MObject("Water"),
    ]
    structure = ModelStructure(PROCESSES, objects, [ElementaryExchange("CO2", MASS)])
    builder = ModelBuilder.from_structure(structure)
    demand = sy.Symbol("demand", positive=True)
    builder.add(
        builder.pull_production("Widget", demand, until_objects=["Slag", "Water"])
    )
    return builder.build(RECIPE), demand


@pytest.fixture(scope="module")
def allocated():
    model, demand = build_toy_model()
    return AllocatedSystem(model, {demand: 100}, RULE)


@pytest.fixture(scope="module")
def structure(allocated):
    return allocated.model.structure


def widget(allocated, expand, **kwargs):
    return allocated.contributions(object="Widget", expand=expand, **kwargs)


class TestIntensities:
    """The hand-computed values the rest of the tests are built on."""

    def test_object_intensities_match_hand_computation(self, allocated):
        mu = allocated.object_intensities["CO2"]
        assert mu["Electricity"] == pytest.approx(0.05)
        assert mu["Ore"] == pytest.approx(0.3)
        assert mu["Steel"] == pytest.approx(0.925)
        assert mu["Slag"] == pytest.approx(0.925)
        assert mu["Widget"] == pytest.approx(4.25)

    def test_unsupplied_object_has_no_intensity(self, allocated):
        assert np.isnan(allocated.object_intensities["CO2"]["Water"])
        assert allocated.unsupplied_objects == ("Water",)

    def test_total_burdens_are_the_intensities_scaled_up(self, allocated):
        # 100 widgets, bearing the whole system's burden between them.
        assert allocated.object_burdens.loc["Widget", "CO2"] == pytest.approx(425.0)
        assert allocated.direct_burdens["CO2"].sum() == pytest.approx(425.0)


class TestFirstTier:
    """Expanding only the object's own producers."""

    @pytest.fixture
    def result(self, allocated, structure):
        return widget(allocated, ProcessSet.producers_of(structure, "Widget"))

    def test_direct_burden_is_the_producer_s_own(self, result):
        assert result.direct["CO2"].to_dict() == pytest.approx({"MakeWidget": 0.4})

    def test_upstream_burden_is_cradle_to_gate_per_input(self, result):
        upstream = result.upstream["CO2"].xs("MakeWidget")
        assert upstream.to_dict() == pytest.approx(
            {"Steel": 1.85, "Slag": 1.85, "Electricity": 0.15, "Water": 0.0}
        )

    def test_contributions_add_up(self, result):
        assert result.check()
        assert result.total["CO2"] == pytest.approx(4.25)

    def test_quantities_report_what_was_taken_in(self, result):
        quantities = result.quantities.set_index("object")["quantity"]
        assert quantities["Steel"] == pytest.approx(2.0)
        assert quantities["Electricity"] == pytest.approx(3.0)


class TestExpandingFurther:
    """Expanding another process opens up what it took in."""

    @pytest.fixture
    def result(self, allocated, structure):
        return widget(
            allocated, ProcessSet.producers_of(structure, "Widget") | "MakeSteel"
        )

    def test_steel_and_slag_are_no_longer_collapsed(self, result):
        assert set(result.upstream.index.get_level_values("object")) == {
            "Electricity",
            "Water",
            "Ore",
        }

    def test_direct_burden_pools_a_process_s_co_products(self, result):
        # MakeSteel is operated twice per widget, once as far as Steel is
        # concerned and once as far as Slag is: 2 x 1.2.
        assert result.direct["CO2"].to_dict() == pytest.approx(
            {"MakeWidget": 0.4, "MakeSteel": 2.4}
        )

    def test_a_shared_input_is_reported_per_consuming_process(self, result):
        electricity = result.upstream["CO2"].xs("Electricity", level="object")
        assert electricity.to_dict() == pytest.approx(
            {"MakeWidget": 0.15, "MakeSteel": 0.4}
        )

    def test_contributions_add_up(self, result):
        assert result.check()


class TestWholeSupplyChain:
    """With everything expanded, all the burden is reported as direct."""

    @pytest.fixture
    def result(self, allocated, structure):
        return widget(allocated, ProcessSet.upstream_of(structure, object="Widget"))

    def test_burden_is_attributed_to_the_processes_that_emitted_it(self, result):
        assert result.direct["CO2"].to_dict() == pytest.approx(
            {
                "MakeWidget": 0.4,
                "MakeSteel": 2.4,
                "SourceOfOre": 0.9,
                "SourceOfElectricity": 0.55,
            }
        )

    def test_only_the_unsupplied_input_is_left(self, result):
        assert result.upstream.index.get_level_values("object").tolist() == ["Water"]

    def test_contributions_add_up(self, result):
        assert result.check()


class TestEverySetAddsUp:
    """The contributions partition the burden whichever processes are expanded."""

    @pytest.mark.parametrize("size", range(len(PROCESSES) + 1))
    def test_every_subset_of_processes(self, allocated, size):
        ids = [p.id for p in PROCESSES]
        for subset in combinations(ids, size):
            expand = ProcessSet.of(subset) | "MakeWidget"
            assert widget(allocated, expand).check()

    def test_amount_scales_the_whole_breakdown(self, allocated, structure):
        expand = ProcessSet.producers_of(structure, "Widget")
        one = widget(allocated, expand)
        ten = widget(allocated, expand, amount=10)
        assert ten.check()
        assert ten.total["CO2"] == pytest.approx(10 * one.total["CO2"])


class TestReferences:
    def test_process_reference_breaks_down_its_own_intensity(self, allocated):
        result = allocated.contributions(
            process="MakeSteel", expand=ProcessSet.of("MakeSteel")
        )
        assert result.check()
        assert result.total["CO2"] == pytest.approx(1.85)
        assert result.upstream["CO2"].xs("MakeSteel").to_dict() == pytest.approx(
            {"Ore": 0.45, "Electricity": 0.2}
        )

    def test_object_from_process_reference_is_that_route_s_intensity(self, allocated):
        result = allocated.contributions(
            object="Steel", process="MakeSteel", expand=ProcessSet.of("MakeSteel")
        )
        assert result.check()
        assert result.total["CO2"] == pytest.approx(0.925)

    def test_single_producer_route_matches_the_average(self, allocated):
        expand = ProcessSet.of("MakeSteel")
        route = allocated.contributions(
            object="Steel", process="MakeSteel", expand=expand
        )
        average = allocated.contributions(object="Steel", expand=expand)
        assert route.total["CO2"] == pytest.approx(average.total["CO2"])


class TestUnsuppliedInputs:
    @pytest.fixture
    def result(self, allocated, structure):
        return widget(allocated, ProcessSet.producers_of(structure, "Widget"))

    def test_reported_with_the_quantity_taken_in(self, result):
        unsupplied = result.unsupplied()
        assert unsupplied["object"].tolist() == ["Water"]
        assert unsupplied["quantity"].tolist() == pytest.approx([0.5])

    def test_carries_no_burden(self, result):
        water = result.table[result.table["object"] == "Water"]
        assert water["kind"].tolist() == ["unsupplied"]
        assert water["CO2"].tolist() == pytest.approx([0.0])

    def test_cannot_break_down_an_object_with_no_supply(self, allocated):
        with pytest.raises(ValueError, match="no modelled supply"):
            allocated.contributions(object="Water", expand=ProcessSet.of("SourceOfOre"))


class TestGrouping:
    @pytest.fixture
    def result(self, allocated, structure):
        return widget(allocated, ProcessSet.producers_of(structure, "Widget"))

    def test_group_inputs_into_categories(self, result):
        grouped = result.with_group(
            "category",
            {"metal": ["Steel", "Slag"], "utilities": ["Electricity", "Water"]},
            direct_label="direct emissions",
        )
        assert grouped.by("category")["CO2"].to_dict() == pytest.approx(
            {"metal": 3.7, "utilities": 0.15, "direct emissions": 0.4}
        )

    def test_grouping_still_adds_up(self, result):
        grouped = result.with_group(
            "category", {"all": ["Steel", "Slag", "Electricity", "Water"]}
        )
        assert grouped.check()

    def test_by_object_keeps_direct_burden_visible(self, result):
        assert result.by("object")["CO2"]["direct"] == pytest.approx(0.4)

    def test_group_may_name_an_id_that_does_not_appear(self, allocated, structure):
        # Steel and Slag are supplied by an expanded process here, so they are
        # not collapsed: naming them is accepted, and they produce no row.
        grouped = widget(
            allocated, ProcessSet.producers_of(structure, "Widget") | "MakeSteel"
        ).with_group(
            "category",
            {
                "metal": ["Steel", "Slag"],
                "raw": ["Ore"],
                "utilities": ["Electricity", "Water"],
            },
        )
        by_category = grouped.by("category")
        assert "metal" not in by_category.index
        assert set(by_category.index) == {"raw", "utilities", "direct"}
        assert grouped.check()

    def test_unknown_group_id_is_rejected(self, result):
        with pytest.raises(ValueError, match="unknown id"):
            result.with_group("category", {"metal": ["Stele"]})

    def test_overlapping_groups_are_rejected(self, result):
        with pytest.raises(ValueError, match="disjoint"):
            result.with_group("category", {"a": ["Steel"], "b": ["Steel", "Slag"]})


class TestProcessSet:
    def test_producers_of(self, structure):
        assert ProcessSet.producers_of(structure, "Steel").processes == {"MakeSteel"}

    def test_tiers_walks_upstream(self, structure):
        assert ProcessSet.tiers(structure, 1, object="Widget").processes == {
            "MakeWidget"
        }
        assert ProcessSet.tiers(structure, 2, object="Widget").processes == {
            "MakeWidget",
            "MakeSteel",
            "SourceOfElectricity",
        }
        assert ProcessSet.tiers(structure, 9, object="Widget").processes == {
            p.id for p in PROCESSES
        }

    def test_tiers_needs_a_starting_point(self, structure):
        with pytest.raises(ValueError, match="exactly one"):
            ProcessSet.tiers(structure, 1)

    def test_upstream_of_stopping_at_objects(self, structure):
        expand = ProcessSet.upstream_of(
            structure, object="Widget", stopping_at_objects=["Electricity"]
        )
        assert expand.processes == {"MakeWidget", "MakeSteel", "SourceOfOre"}

    def test_upstream_of_stopping_at_processes(self, structure):
        # Electricity is still reached, because MakeWidget uses it directly.
        expand = ProcessSet.upstream_of(
            structure, object="Widget", stopping_at_processes=["MakeSteel"]
        )
        assert expand.processes == {"MakeWidget", "SourceOfElectricity"}

    def test_suppliers_are_the_processes_just_outside(self, structure):
        expand = ProcessSet.producers_of(structure, "Widget")
        assert expand.suppliers(structure) == {"MakeSteel", "SourceOfElectricity"}

    def test_external_inputs_include_unsupplied_ones(self, structure):
        expand = ProcessSet.producers_of(structure, "Widget")
        assert expand.external_inputs(structure) == {
            "Steel",
            "Slag",
            "Electricity",
            "Water",
        }

    def test_combining_sets(self, structure):
        makers = ProcessSet.producers_of(structure, "Widget")
        assert (makers | "MakeSteel").processes == {"MakeWidget", "MakeSteel"}
        assert ((makers | "MakeSteel") - "MakeSteel").processes == {"MakeWidget"}

    def test_behaves_like_a_set(self, structure):
        expand = ProcessSet.producers_of(structure, "Widget") | "MakeSteel"
        assert len(expand) == 2
        assert "MakeSteel" in expand
        assert sorted(expand) == ["MakeSteel", "MakeWidget"]


class TestErrors:
    def test_no_reference(self, allocated):
        with pytest.raises(ValueError, match="object=, process=, or both"):
            allocated.contributions(expand=ProcessSet.of("MakeWidget"))

    def test_unknown_object(self, allocated):
        with pytest.raises(ValueError, match="Unknown object"):
            allocated.contributions(
                object="Nonsense", expand=ProcessSet.of("MakeWidget")
            )

    def test_process_not_expanded(self, allocated):
        with pytest.raises(ValueError, match="not being expanded"):
            allocated.contributions(
                process="MakeSteel", expand=ProcessSet.of("MakeWidget")
            )

    def test_process_does_not_produce_the_object(self, allocated):
        with pytest.raises(ValueError, match="does not produce"):
            allocated.contributions(
                object="Widget",
                process="MakeSteel",
                expand=ProcessSet.of("MakeSteel"),
            )


# ============================================================================
# A loop: recycled product re-enters its own supply chain.
# ============================================================================


def build_loop_model():
    processes = [
        Process("Virgin", produces=["Product"], consumes=["Raw"], exchanges=["CO2"]),
        Process("Use", produces=["Waste"], consumes=["Product"]),
        Process(
            "Recycling", produces=["Product"], consumes=["Waste"], exchanges=["CO2"]
        ),
        Process("MineRaw", produces=["Raw"], consumes=[], exchanges=["CO2"]),
    ]
    objects = [
        MObject("Raw", has_market=True),
        MObject("Product", has_market=True),
        MObject("Waste", has_market=True),
    ]
    structure = ModelStructure(processes, objects, [ElementaryExchange("CO2", MASS)])
    builder = ModelBuilder.from_structure(structure)
    demand = sy.Symbol("demand", positive=True)
    builder.add(
        builder.pull_process_output("Recycling", "Product", demand * sy.Rational(1, 2))
    )
    builder.add(
        builder.pull_process_output("Virgin", "Product", demand * sy.Rational(1, 2))
    )
    builder.add(builder.push_consumption("Product", demand))
    builder.add(
        builder.push_process_input(
            "Recycling", "Waste", builder.object_consumption_deficit("Waste")
        )
    )
    recipe = {
        "Virgin": {
            "consumes": {"Raw": 1.0},
            "produces": {"Product": 1.0},
            "exchanges": {"CO2": 2.0},
        },
        "Use": {"consumes": {"Product": 1.0}, "produces": {"Waste": 1.0}},
        "Recycling": {
            "consumes": {"Waste": 1.0},
            "produces": {"Product": 1.0},
            "exchanges": {"CO2": 0.5},
        },
        "MineRaw": {"produces": {"Raw": 1.0}, "exchanges": {"CO2": 10.0}},
    }
    return builder.build(recipe), demand


@pytest.fixture(scope="module")
def loop():
    model, demand = build_loop_model()
    scope = Scope(waste_input_burden="propagate")
    return AllocatedSystem(model, {demand: 100}, ByValue(), scope=scope)


class TestLoop:
    """A product whose own waste feeds back into making it."""

    @pytest.mark.parametrize(
        "expanded",
        [
            ("Virgin", "Recycling"),
            ("Virgin", "Recycling", "Use"),
            ("Virgin", "Recycling", "Use", "MineRaw"),
            ("Virgin",),
        ],
    )
    def test_contributions_add_up_through_the_loop(self, loop, expanded):
        result = loop.contributions(object="Product", expand=ProcessSet.of(expanded))
        assert result.check()

    def test_tiers_terminates(self, loop):
        expand = ProcessSet.tiers(loop.model.structure, 5, object="Product")
        assert expand.processes == {"Virgin", "Recycling", "Use", "MineRaw"}


# ============================================================================
# Scope and rules interact with contributions the same way as with the solve.
# ============================================================================


class TestScopeAndRules:
    def test_excluded_processes_are_never_expanded(self):
        model, demand = build_toy_model()
        allocated = AllocatedSystem(
            model,
            {demand: 100},
            RULE,
            scope=Scope(excluded_processes=frozenset({"SourceOfOre"})),
        )
        result = allocated.contributions(
            object="Widget",
            expand=ProcessSet.upstream_of(model.structure, object="Widget"),
        )
        assert "SourceOfOre" not in result.direct.index
        assert result.check()

    def test_wastes_shorthand_leaves_the_waste_carrying_nothing(self):
        model, demand = build_loop_model()
        allocated = AllocatedSystem(model, {demand: 100}, ByValue(), wastes={"Waste"})
        result = allocated.contributions(
            object="Product", expand=ProcessSet.of(("Virgin", "Recycling"))
        )
        assert result.check()
        waste = result.table[result.table["object"] == "Waste"]
        assert waste.empty or waste["CO2"].sum() == pytest.approx(0.0)
