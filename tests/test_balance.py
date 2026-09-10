"""Tests for market balance tracking (`flowprog.balance`)."""

import pytest
import sympy as sy
from hypothesis import given, settings, strategies as st
from rdflib import URIRef

from flowprog import ModelBuilder, Process, Object, merge_activities
from flowprog.balance import Effect, Sign, Verdict, sign_of


MASS = URIRef("http://qudt.org/vocab/quantitykind/Mass")


def MObject(id, *args, **kwargs):
    return Object(id, MASS, *args, **kwargs)


D = sy.Symbol("D", positive=True)


def chain_builder():
    """in -> P1 -> mid (market) -> P2 -> out"""
    return ModelBuilder(
        [
            Process("P1", consumes=["in"], produces=["mid"]),
            Process("P2", consumes=["mid"], produces=["out"]),
        ],
        [MObject("in"), MObject("mid", has_market=True), MObject("out")],
    )


def merit_order_builder(with_backstop, capacities=("C1", "C2")):
    """Demand for a service met by two capacity-limited suppliers of an
    intermediate, optionally with an unlimited supplier behind them."""
    builder = ModelBuilder(
        [
            Process("Cheap", consumes=["Fuel"], produces=["Elec"]),
            Process("Mid", consumes=["Fuel"], produces=["Elec"]),
            Process("Backstop", consumes=["Fuel"], produces=["Elec"]),
            Process("Use", consumes=["Elec"], produces=["Service"]),
        ],
        [MObject("Fuel"), MObject("Elec", has_market=True), MObject("Service")],
    )
    builder.add(
        builder.pull_production("Service", D, until_objects=["Elec"]), label="demand"
    )
    for process_id, name in zip(("Cheap", "Mid"), capacities):
        proposal = builder.pull_process_output(
            process_id, "Elec", builder.object_production_deficit("Elec")
        )
        builder.add(
            builder.limit(
                proposal,
                builder.expr("ProcessOutput", process_id=process_id, object_id="Elec"),
                sy.Symbol(name, positive=True),
            ),
            label=f"dispatch {process_id}",
        )
    if with_backstop:
        builder.add(
            builder.pull_process_output(
                "Backstop", "Elec", builder.object_production_deficit("Elec")
            ),
            label="backstop",
        )
    return builder


class TestSignOf:
    def test_recipe_and_activity_symbols_are_non_negative(self):
        structure = chain_builder().structure
        assert sign_of(structure.S[1, 0]) is Sign.NON_NEGATIVE
        assert sign_of(structure.Y[0] * structure.S[1, 0]) is Sign.NON_NEGATIVE
        assert sign_of(-structure.Y[0] * structure.S[1, 0]) is Sign.NON_POSITIVE

    def test_sums_of_mixed_sign_are_unknown(self):
        structure = chain_builder().structure
        assert sign_of(structure.Y[0] - structure.Y[1]) is Sign.UNKNOWN

    def test_zero(self):
        assert sign_of(sy.S.Zero) is Sign.ZERO

    def test_max_with_zero_is_non_negative(self):
        assert sign_of(sy.Max(0, sy.Symbol("x"), evaluate=False)) is Sign.NON_NEGATIVE

    def test_follows_intermediate_definitions(self):
        x = sy.Symbol("x")
        y = sy.Symbol("y", positive=True)
        assert sign_of(x, {x: -y}) is Sign.NON_POSITIVE

    def test_unknown_symbol_is_unknown(self):
        assert sign_of(sy.Symbol("p")) is Sign.UNKNOWN


class TestProvedBalanced:
    def test_propagation_that_is_not_cut_short_balances(self):
        builder = chain_builder()
        builder.add(builder.pull_production("out", D))
        trace = builder.build().balance_trace

        assert trace.residual("mid") == 0
        assert trace.verdict("mid")[0] is Verdict.BALANCED

    def test_deficit_step_closes_what_until_objects_left_open(self):
        builder = chain_builder()
        builder.add(
            builder.pull_production("out", D, until_objects=["mid"]), label="demand"
        )
        builder.add(
            builder.pull_production("mid", builder.object_production_deficit("mid")),
            label="supply",
        )
        trace = builder.build().balance_trace

        assert trace.verdict("mid")[0] is Verdict.BALANCED
        assert [e.effect for e in trace.events_for("mid")] == [Effect.OPENS, Effect.CLOSES]

    def test_capacity_limited_options_followed_by_an_unlimited_one(self):
        """The usual dispatch pattern: whatever the capacities turn out to be,
        the last unlimited supplier takes up the rest."""
        trace = merit_order_builder(with_backstop=True).build().balance_trace

        assert trace.verdict("Elec")[0] is Verdict.BALANCED
        assert [e.effect for e in trace.events_for("Elec")] == [
            Effect.OPENS,
            Effect.CLOSES_UNLESS_CONSTRAINED,
            Effect.CLOSES_UNLESS_CONSTRAINED,
            Effect.CLOSES,
        ]

    def test_object_without_a_market_is_not_tracked(self):
        builder = chain_builder()
        builder.add(builder.pull_production("out", D))
        trace = builder.build().balance_trace

        assert trace.markets() == ["mid"]


class TestOpen:
    def test_propagation_cut_short_and_never_picked_up(self):
        builder = chain_builder()
        builder.add(
            builder.pull_production("out", D, until_objects=["mid"]), label="demand"
        )
        model = builder.build()
        trace = model.balance_trace

        verdict, reason = trace.verdict("mid")
        assert verdict is Verdict.OPEN
        assert "never balances" in reason
        # The residual says how much is missing, in the model's own terms
        assert trace.residual("mid") == -D * model.U[1, 1] / model.S[2, 1]

    def test_pushing_through_a_process_with_stock_leaves_the_output_unsupplied(self):
        """Pushing into a process that can accumulate stock propagates onwards
        without recording the process's own output, so the object in between is
        consumed without being produced."""
        builder = ModelBuilder(
            [
                Process("Use", consumes=["New"], produces=["Old"], has_stock=True),
                Process("Treat", consumes=["Old"], produces=["Waste"]),
            ],
            [MObject("New"), MObject("Old", has_market=True), MObject("Waste")],
        )
        builder.add(builder.push_consumption("New", D), label="products into use")
        trace = builder.build().balance_trace

        assert trace.verdict("Old")[0] is Verdict.OPEN


class TestConditional:
    def test_capacity_limited_options_with_nothing_behind_them(self):
        """Demand beyond the total capacity has nowhere to come from."""
        trace = merit_order_builder(with_backstop=False).build().balance_trace

        verdict, reason = trace.verdict("Elec")
        assert verdict is Verdict.CONDITIONAL
        assert "demand can go unmet" in reason
        assert len(trace.breakpoints("Elec")) == 2

    def test_minimum_operating_threshold(self):
        """A supplier that must either run above a minimum or not at all cannot
        meet demand below that minimum."""
        builder = ModelBuilder(
            [
                Process("Src", consumes=["Fuel"], produces=["Elec"]),
                Process("Use", consumes=["Elec"], produces=["Service"]),
            ],
            [MObject("Fuel"), MObject("Elec", has_market=True), MObject("Service")],
        )
        builder.add(
            builder.pull_production("Service", D, until_objects=["Elec"]),
            label="demand",
        )
        proposal = builder.pull_process_output(
            "Src", "Elec", builder.object_production_deficit("Elec")
        )
        builder.add(
            builder.floor(
                proposal,
                builder.expr("ProcessOutput", process_id="Src", object_id="Elec"),
                sy.Symbol("minimum", positive=True),
            ),
            label="supply",
        )
        trace = builder.build().balance_trace

        assert trace.verdict("Elec")[0] is Verdict.CONDITIONAL
        assert trace.breakpoints("Elec")

    def test_a_deficit_split_by_share_parameters_is_not_proved(self):
        """Splitting what is missing between routes closes the market only if
        the shares sum to one, which is a claim about parameter values."""
        builder = ModelBuilder(
            [
                Process("RouteA", consumes=["Ore"], produces=["Metal"]),
                Process("RouteB", consumes=["Scrap"], produces=["Metal"]),
                Process("Fab", consumes=["Metal"], produces=["Part"]),
            ],
            [
                MObject("Ore"),
                MObject("Scrap"),
                MObject("Metal", has_market=True),
                MObject("Part"),
            ],
        )
        share_a, share_b = sy.symbols("share_a share_b", positive=True)
        builder.add(
            builder.pull_production("Part", D, until_objects=["Metal"]), label="demand"
        )
        deficit = builder.object_production_deficit("Metal")
        # Merged into one step, so both shares are of the same shortfall
        builder.add(
            merge_activities(
                builder.pull_process_output("RouteA", "Metal", deficit * share_a),
                builder.pull_process_output("RouteB", "Metal", deficit * share_b),
            ),
            label="supply",
        )
        trace = builder.build().balance_trace

        assert trace.verdict("Metal")[0] is not Verdict.BALANCED

    def test_shares_that_are_known_to_sum_to_one_do_balance(self):
        """The same model with numeric shares: sympy can see the cancellation."""
        builder = ModelBuilder(
            [
                Process("RouteA", consumes=["Ore"], produces=["Metal"]),
                Process("RouteB", consumes=["Scrap"], produces=["Metal"]),
                Process("Fab", consumes=["Metal"], produces=["Part"]),
            ],
            [
                MObject("Ore"),
                MObject("Scrap"),
                MObject("Metal", has_market=True),
                MObject("Part"),
            ],
        )
        builder.add(
            builder.pull_production("Part", D, until_objects=["Metal"]), label="demand"
        )
        deficit = builder.object_production_deficit("Metal")
        builder.add(
            merge_activities(
                builder.pull_process_output(
                    "RouteA", "Metal", deficit * sy.Rational(1, 4)
                ),
                builder.pull_process_output(
                    "RouteB", "Metal", deficit * sy.Rational(3, 4)
                ),
            ),
            label="supply",
        )
        trace = builder.build().balance_trace

        assert trace.verdict("Metal")[0] is Verdict.BALANCED


class TestReporting:
    def test_explain_names_the_steps_involved(self):
        trace = merit_order_builder(with_backstop=True).build().balance_trace
        text = trace.explain("Elec")

        assert "Elec: balanced" in text
        assert "demand" in text
        assert "dispatch Cheap" in text

    def test_dataframe_has_a_row_per_market(self):
        trace = merit_order_builder(with_backstop=False).build().balance_trace
        table = trace.to_dataframe()

        assert list(table.columns) == ["object", "verdict", "reason", "residual"]
        assert list(table["object"]) == ["Elec"]

    def test_open_market_is_listed(self):
        builder = chain_builder()
        builder.add(builder.pull_production("out", D, until_objects=["mid"]))
        trace = builder.build().balance_trace

        assert trace.markets(Verdict.OPEN) == ["mid"]

    def test_markets_can_be_restricted_to_one_verdict(self):
        trace = merit_order_builder(with_backstop=True).build().balance_trace

        assert trace.markets() == ["Elec"]
        assert trace.markets(Verdict.BALANCED) == ["Elec"]
        assert trace.markets(Verdict.OPEN) == []

    def test_saving_and_loading_keeps_what_was_found(self, tmp_path):
        from flowprog import SympyModel

        builder = merit_order_builder(with_backstop=True)
        path = tmp_path / "model.json"
        builder.build().save(str(path))
        trace = SympyModel.load(str(path)).balance_trace

        assert trace.verdict("Elec")[0] is Verdict.BALANCED
        assert [e.effect for e in trace.events_for("Elec")] == [
            Effect.OPENS,
            Effect.CLOSES_UNLESS_CONSTRAINED,
            Effect.CLOSES_UNLESS_CONSTRAINED,
            Effect.CLOSES,
        ]

    def test_a_model_built_without_step_history_still_resolves(self):
        """Balances worked out from the accumulated activities are the same
        quantity, just not reduced, so nothing can be proved from them."""
        from flowprog import SympyModel

        builder = chain_builder()
        builder.add(builder.pull_production("out", D))
        compiled = builder.build()
        recipe = {
            compiled.U[0, 0]: 1.0,
            compiled.S[1, 0]: 1.0,
            compiled.U[1, 1]: 1.0,
            compiled.S[2, 1]: 1.0,
        }
        plain = SympyModel(
            compiled.structure,
            values=dict(compiled._values),
            intermediates=compiled._intermediates,
            recipe_data=recipe,
        )

        assert compiled.balance_trace.verdict("mid")[0] is Verdict.BALANCED
        assert plain.balance_trace.verdict("mid")[0] is not Verdict.BALANCED
        # ... but it is still the same quantity, and evaluates to zero
        value = plain.eval(plain.structure.Balance[1], {D: 10.0})
        assert float(value) == pytest.approx(0)


class TestAgreesWithTheNumbers:
    """The whole point is that a market proved to balance really does, and one
    reported as not balancing really can be out of balance."""

    RECIPE_KEYS = ((0, 0), (1, 0), (1, 1), (2, 1))

    @given(
        st.floats(min_value=0, max_value=1e6),
        st.floats(min_value=0, max_value=1e6),
        st.floats(min_value=0, max_value=1e6),
    )
    @settings(deadline=None, max_examples=50)
    def test_balanced_market_is_balanced_at_any_parameter_values(
        self, demand, capacity_1, capacity_2
    ):
        builder = merit_order_builder(with_backstop=True)
        model = builder.build(
            {
                builder.U[0, 0]: 2.0,
                builder.S[1, 0]: 1.0,
                builder.U[0, 1]: 2.5,
                builder.S[1, 1]: 1.0,
                builder.U[0, 2]: 3.0,
                builder.S[1, 2]: 1.0,
                builder.U[1, 3]: 1.0,
                builder.S[2, 3]: 1.0,
            }
        )
        assert model.balance_trace.verdict("Elec")[0] is Verdict.BALANCED

        evaluate = model.lambdify(
            expressions={
                "balance": builder.object_balance("Elec"),
                "production": builder.expr("SoldProduction", object_id="Elec"),
            },
            modules="math",
        )
        result = evaluate({"D": demand, "C1": capacity_1, "C2": capacity_2})
        scale = max(result["production"], 1.0)
        assert abs(result["balance"]) < 1e-9 * scale

    def test_conditional_market_really_does_go_out_of_balance(self):
        builder = merit_order_builder(with_backstop=False)
        model = builder.build(
            {
                builder.U[0, 0]: 2.0,
                builder.S[1, 0]: 1.0,
                builder.U[0, 1]: 2.5,
                builder.S[1, 1]: 1.0,
                builder.U[1, 3]: 1.0,
                builder.S[2, 3]: 1.0,
            }
        )
        assert model.balance_trace.verdict("Elec")[0] is Verdict.CONDITIONAL

        evaluate = model.lambdify(
            expressions={"balance": builder.object_balance("Elec")}, modules="math"
        )
        within_capacity = {"D": 50.0, "C1": 60.0, "C2": 50.0}
        beyond_capacity = {"D": 150.0, "C1": 60.0, "C2": 50.0}
        assert evaluate(within_capacity)["balance"] == pytest.approx(0)
        assert evaluate(beyond_capacity)["balance"] == pytest.approx(-40.0)

    def test_tracked_residual_matches_the_model(self):
        """The tracked balance is the same quantity the model itself reports."""
        builder = merit_order_builder(with_backstop=False)
        recipe = {
            builder.U[0, 0]: 2.0,
            builder.S[1, 0]: 1.0,
            builder.U[0, 1]: 2.5,
            builder.S[1, 1]: 1.0,
            builder.U[1, 3]: 1.0,
            builder.S[2, 3]: 1.0,
        }
        model = builder.build(recipe)
        values = {D: 150.0, sy.Symbol("C1", positive=True): 60.0,
                  sy.Symbol("C2", positive=True): 50.0}

        from_model = model.eval(builder.object_balance("Elec"), values)
        from_trace = model.eval_intermediates(
            model.balance_trace.residual("Elec"), values
        ).xreplace(recipe).xreplace(values)
        assert float(from_model) == pytest.approx(float(from_trace))
