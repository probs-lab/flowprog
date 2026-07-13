"""Tests for elementary exchanges: the B matrix and ElementaryBalance.

Elementary exchanges represent flows to/from the environment (CO2, CH4, ...),
as distinct from technosphere objects (S/U matrices).
"""

import pytest
import sympy as sy
from rdflib import URIRef

from flowprog import ModelBuilder, Process, Object, ElementaryExchange

MASS = URIRef("http://qudt.org/vocab/quantitykind/Mass")


def MObject(id, *args, **kwargs):
    return Object(id, MASS, *args, **kwargs)


def MExchange(id, *args, **kwargs):
    return ElementaryExchange(id, MASS, *args, **kwargs)


class TestDeclarationAndLookup:
    def test_lookup_exchange_returns_index(self):
        processes = [Process("P1", produces=["out"], consumes=[])]
        objects = [MObject("out")]
        exchanges = [MExchange("CO2"), MExchange("CH4")]
        builder = ModelBuilder(processes, objects, exchanges)

        assert builder.structure.lookup_exchange("CO2") == 0
        assert builder.structure.lookup_exchange("CH4") == 1

    def test_unknown_exchange_raises(self):
        processes = [Process("P1", produces=["out"], consumes=[])]
        objects = [MObject("out")]
        builder = ModelBuilder(processes, objects, [MExchange("CO2")])

        with pytest.raises(ValueError):
            builder.structure.lookup_exchange("N2O")

    def test_process_declaring_unknown_exchange_raises(self):
        processes = [
            Process("P1", produces=["out"], consumes=[], exchanges=["N2O"])
        ]
        objects = [MObject("out")]
        with pytest.raises(ValueError, match="N2O"):
            ModelBuilder(processes, objects, [MExchange("CO2")])

    def test_recipe_exchange_must_be_declared_id_format(self):
        # B sparsity is structural: recipe values are only accepted for
        # declared (exchange, process) cells, mirroring produces/consumes.
        processes = [Process("P1", produces=["out"], consumes=[])]
        objects = [MObject("out")]
        builder = ModelBuilder(processes, objects, [MExchange("CO2")])
        with pytest.raises(ValueError, match="only lists"):
            builder.build({"P1": {"produces": {"out": 1.0}, "exchanges": {"CO2": 2.0}}})

    def test_recipe_exchange_must_be_declared_symbol_format(self):
        processes = [Process("P1", produces=["out"], consumes=[])]
        objects = [MObject("out")]
        builder = ModelBuilder(processes, objects, [MExchange("CO2")])
        with pytest.raises(ValueError, match="only lists"):
            builder.build({builder.S[0, 0]: 1.0, builder.B[0, 0]: 2.0})

    def test_no_exchanges_declared_by_default(self):
        processes = [Process("P1", produces=["out"], consumes=[])]
        objects = [MObject("out")]
        builder = ModelBuilder(processes, objects)

        assert builder.structure.elementary_exchanges == ()
        assert builder.structure.B.shape == (0, 1)


class TestBMatrix:
    def test_B_shape(self):
        processes = [
            Process("P1", produces=["out"], consumes=[]),
            Process("P2", produces=["out2"], consumes=[]),
        ]
        objects = [MObject("out"), MObject("out2")]
        exchanges = [MExchange("CO2"), MExchange("CH4"), MExchange("N2O")]
        builder = ModelBuilder(processes, objects, exchanges)

        assert builder.structure.B.shape == (3, 2)

    def test_B_has_no_nonnegative_assumption(self):
        """Unlike S/U, B is signed: no nonnegative=True assumption."""
        processes = [Process("P1", produces=["out"], consumes=[])]
        objects = [MObject("out")]
        builder = ModelBuilder(processes, objects, [MExchange("CO2")])

        assert builder.structure.B[0, 0].is_nonnegative is not True
        # Contrast with S, which is declared nonnegative
        assert builder.structure.S[0, 0].is_nonnegative is True

    def test_ElementaryBalance_shape(self):
        processes = [Process("P1", produces=["out"], consumes=[])]
        objects = [MObject("out")]
        exchanges = [MExchange("CO2"), MExchange("CH4")]
        builder = ModelBuilder(processes, objects, exchanges)

        assert builder.structure.ElementaryBalance.shape == (2,)


class TestExprRoles:
    @pytest.fixture
    def m(self):
        processes = [
            Process("P1", produces=["out"], consumes=[], exchanges=["CO2"]),
            Process("P2", produces=["out"], consumes=[], exchanges=["CO2"]),
        ]
        objects = [MObject("out", has_market=True)]
        exchanges = [MExchange("CO2")]
        return ModelBuilder(processes, objects, exchanges)

    def test_process_elementary_flow(self, m):
        expr = m.expr("ProcessElementaryFlow", exchange_id="CO2", process_id="P1")
        assert expr == m.B[0, 0] * m.Y[0]

    def test_elementary_flows_sums_declaring_processes(self, m):
        # Both P1 and P2 declare CO2, so both contribute.
        expr = m.expr("ElementaryFlows", exchange_id="CO2")
        assert expr == m.B[0, 0] * m.Y[0] + m.B[0, 1] * m.Y[1]

    def test_elementary_flows_excludes_non_declaring_process(self):
        # A process that does not declare the exchange contributes nothing.
        processes = [
            Process("P1", produces=["out"], consumes=[], exchanges=["CO2"]),
            Process("P2", produces=["out"], consumes=[]),
        ]
        m = ModelBuilder(processes, [MObject("out", has_market=True)], [MExchange("CO2")])
        expr = m.expr("ElementaryFlows", exchange_id="CO2")
        assert expr == m.B[0, 0] * m.Y[0]

    def test_elementary_flows_limit_to_processes(self, m):
        expr = m.expr(
            "ElementaryFlows", exchange_id="CO2", limit_to_processes={"P2"}
        )
        assert expr == m.B[0, 1] * m.Y[1]


class TestEvalUnexpanded:
    """SympyModel.eval(expr, expand_intermediates=False): the raw, lazy path --
    resolves structural + recipe symbols but leaves intermediates un-expanded,
    for a later batched lambdify()."""

    def _model(self):
        processes = [
            Process("Dirty", produces=["out"], consumes=[], exchanges=["CO2"]),
            Process("Clean", produces=["out"], consumes=[], exchanges=["CO2"]),
            Process("Consumer", produces=["product"], consumes=["out"])
        ]
        objects = [MObject("out", has_market=True), MObject("product")]
        builder = ModelBuilder(processes, objects, [MExchange("CO2")])
        # Build a model that has intermediates
        builder.add(
            builder.pull_production(
                "product",
                sy.S(15),
                allocate_backwards={"out": {"Dirty": 2./3, "Clean": 1./3}}
            )
        )
        model = builder.build(
            {
                "Dirty": {"produces": {"out": 1.0}, "exchanges": {"CO2": 2.0}},
                "Clean": {"produces": {"out": 1.0}, "exchanges": {"CO2": -0.5}},
                "Consumer": {"produces": {"product": 1.0}, "consumes": {"out": 1.0}},
            }
        )
        assert len(model._intermediates) > 0
        return model

    def test_matches_expanded_numerically(self):
        model = self._model()
        structural = model.expr("ElementaryFlows", exchange_id="CO2")
        raw = model.eval(structural, expand_intermediates=False)
        func = model.lambdify(expressions={"total": raw})
        assert func({})["total"] == pytest.approx(float(model.eval(structural)))
        assert float(model.eval(structural)) == pytest.approx(2.0 * 10 + -0.5 * 5)

    def test_leaves_recipe_symbols_for_lambdify(self):
        """expand_intermediates=False substitutes raw accumulated Y[j] and
        recipe B[e,j] values but does not expand intermediates -- ready for a
        single batched lambdify(), same contract as the structural
        elementary_flow_table()."""
        model = self._model()
        raw = model.eval(
            model.expr("ElementaryFlows", exchange_id="CO2"),
            expand_intermediates=False,
        )
        assert sy.Symbol("x1") in raw.free_symbols

    def test_limit_to_processes(self):
        model = self._model()
        raw = model.eval(
            model.expr(
                "ElementaryFlows", exchange_id="CO2", limit_to_processes={"Clean"}
            ),
            expand_intermediates=False,
        )
        func = model.lambdify(expressions={"total": raw})
        assert func({})["total"] == pytest.approx(-0.5 * 5)

    def test_non_declaring_process_absent_from_flows(self):
        """A process that does not declare the exchange contributes nothing
        in the un-expanded path -- no dangling B[e,j] symbol (structural
        sparsity, unlike a declared-but-unvalued cell which would persist)."""
        processes = [
            Process("Dirty", produces=["out"], consumes=[], exchanges=["CO2"]),
            Process("Clean", produces=["out"], consumes=[]),
        ]
        objects = [MObject("out", has_market=True)]
        builder = ModelBuilder(processes, objects, [MExchange("CO2")])
        builder.add({builder.Y[0]: 10, builder.Y[1]: 5})
        # Only "Dirty" declares CO2; "Clean" does not.
        model = builder.build(
            {
                "Dirty": {"produces": {"out": 1.0}, "exchanges": {"CO2": 2.0}},
                "Clean": {"produces": {"out": 1.0}},
            }
        )

        raw = model.eval(
            model.expr("ElementaryFlows", exchange_id="CO2"),
            expand_intermediates=False,
        )
        assert not raw.atoms(sy.Indexed)
        func = model.lambdify(expressions={"total": raw})
        assert func({})["total"] == pytest.approx(2.0 * 10)

    def test_values_substituted_into_visible_expression(self):
        """`values` may be combined with expand_intermediates=False: they are
        substituted into the visible expression, chaining through a recipe
        value that is itself a parameter (B[e,j] -> EF -> number)."""
        processes = [
            Process("Dirty", produces=["out"], consumes=[], exchanges=["CO2"])
        ]
        objects = [MObject("out", has_market=True)]
        builder = ModelBuilder(processes, objects, [MExchange("CO2")])
        builder.add({builder.Y[0]: 10})
        ef = sy.Symbol("EF")
        model = builder.build(
            {"Dirty": {"produces": {"out": 1.0}, "exchanges": {"CO2": ef}}}
        )
        raw = model.eval(
            model.expr("ElementaryFlows", exchange_id="CO2"),
            values={ef: 3.0},
            expand_intermediates=False,
        )
        assert raw == pytest.approx(10 * 3.0)


class TestElementaryBalance:
    def test_returns_indexed_structural_symbol(self):
        processes = [Process("P1", produces=["out"], consumes=[])]
        objects = [MObject("out")]
        builder = ModelBuilder(processes, objects, [MExchange("CO2")])

        result = builder.elementary_balance("CO2")
        assert isinstance(result, sy.Indexed)
        assert result.base == builder.structure.ElementaryBalance

    def test_unknown_exchange_raises(self):
        processes = [Process("P1", produces=["out"], consumes=[])]
        objects = [MObject("out")]
        builder = ModelBuilder(processes, objects, [MExchange("CO2")])

        with pytest.raises(ValueError):
            builder.elementary_balance("N2O")

    def test_resolves_against_accumulated_Y(self):
        """ElementaryBalance[e] resolves to Sum_j B[e,j] * Y[j] against accumulated state."""
        processes = [
            Process("Dirty", produces=["out"], consumes=[], exchanges=["CO2"]),
            Process("Clean", produces=["out"], consumes=[], exchanges=["CO2"]),
        ]
        objects = [MObject("out", has_market=True)]
        builder = ModelBuilder(processes, objects, [MExchange("CO2")])

        builder.add({builder.Y[0]: 10, builder.Y[1]: 5})
        model = builder.build(
            {
                "Dirty": {"produces": {"out": 1.0}, "exchanges": {"CO2": 2.0}},
                "Clean": {"produces": {"out": 1.0}, "exchanges": {"CO2": -0.5}},
            }
        )

        resolved = model.eval(builder.elementary_balance("CO2"))
        assert resolved == pytest.approx(2.0 * 10 + -0.5 * 5)

    def test_undeclared_exchange_contributes_zero(self):
        """A process that does not declare an exchange contributes nothing to
        its balance -- the cell is structurally absent, not a free symbol."""
        processes = [Process("P1", produces=["out"], consumes=[])]
        objects = [MObject("out", has_market=True)]
        builder = ModelBuilder(processes, objects, [MExchange("CO2")])

        builder.add({builder.Y[0]: 10})
        model = builder.build({"P1": {"produces": {"out": 1.0}}})

        resolved = model.eval(builder.elementary_balance("CO2"))
        assert resolved == 0

    def test_declared_but_unvalued_B_persists_as_symbol(self):
        """A declared exchange with no recipe value persists as a symbol,
        exactly like S/U -- it is not silently zeroed."""
        processes = [Process("P1", produces=["out"], consumes=[], exchanges=["CO2"])]
        objects = [MObject("out", has_market=True)]
        builder = ModelBuilder(processes, objects, [MExchange("CO2")])

        builder.add({builder.Y[0]: 10})
        model = builder.build({"P1": {"produces": {"out": 1.0}}})

        resolved = model.eval(builder.elementary_balance("CO2"))
        # B[0, 0] has no recipe value, so 10 * B[0, 0] persists
        assert resolved == 10 * builder.B[0, 0]


class TestElementaryBalanceMidBuildLimit:
    """Demonstrate emissions appearing in a limit expression. Cumulative CO2
    switches supply between two processes.
    """

    def _build(self):
        processes = [
            Process("Dirty", produces=["out"], consumes=[], exchanges=["CO2"]),
            Process("Clean", produces=["out"], consumes=[], exchanges=["CO2"]),
        ]
        objects = [MObject("out", has_market=True)]
        exchanges = [MExchange("CO2")]
        builder = ModelBuilder(processes, objects, exchanges)

        demand = sy.Symbol("demand", nonnegative=True)
        cap = sy.Symbol("cap", nonnegative=True)

        dirty_proposal = builder.pull_process_output("Dirty", "out", demand)
        dirty_limited = builder.limit(dirty_proposal, builder.elementary_balance("CO2"), cap)
        builder.add(dirty_limited, label="dirty supply, capped by cumulative CO2")

        remaining = demand - builder.object_balance("out")
        builder.add(
            builder.pull_process_output("Clean", "out", remaining),
            label="clean makes up the remainder",
        )

        recipe = {
            builder.S[0, 0]: 1.0,
            builder.S[0, 1]: 1.0,
            builder.B[0, 0]: 2.0,
            builder.B[0, 1]: 0.0,
        }
        return builder, builder.build(recipe), demand, cap

    def test_below_cap_all_from_dirty(self):
        builder, model, demand, cap = self._build()
        data = {demand: 10, cap: 100}
        y_dirty = model.eval(model.Y[0]).subs(data)
        y_clean = model.eval(model.Y[1]).subs(data)
        assert float(y_dirty) == pytest.approx(10)
        assert float(y_clean) == pytest.approx(0)

    def test_above_cap_regime_boundary_at_cap(self):
        builder, model, demand, cap = self._build()
        data = {demand: 100, cap: 40}
        y_dirty = model.eval(model.Y[0]).subs(data)
        y_clean = model.eval(model.Y[1]).subs(data)

        # Dirty is capped so that its cumulative CO2 lands exactly at the cap
        co2 = model.eval(model.structure.ElementaryBalance[0]).subs(data)
        assert float(co2) == pytest.approx(40)
        assert float(y_dirty) == pytest.approx(20)  # 40 tCO2 / 2 tCO2 per unit
        # Clean makes up the rest so total production still matches demand
        assert float(y_dirty) + float(y_clean) == pytest.approx(100)


class TestRecipeData:
    """B values are recipe data, set alongside S and U."""

    def _builder(self):
        processes = [
            Process("Dirty", produces=["out"], consumes=[], exchanges=["CO2"]),
            Process("Clean", produces=["out"], consumes=[], exchanges=["CO2"]),
        ]
        objects = [MObject("out", has_market=True)]
        exchanges = [MExchange("CO2")]
        return ModelBuilder(processes, objects, exchanges)

    def test_symbol_format_sets_B(self):
        builder = self._builder()
        builder.add({builder.Y[0]: 10, builder.Y[1]: 5})
        model = builder.build(
            {
                builder.S[0, 0]: 1.0,
                builder.S[0, 1]: 1.0,
                builder.B[0, 0]: 2.0,
                builder.B[0, 1]: -0.5,
            }
        )
        assert model.eval(model.structure.ElementaryBalance[0]) == pytest.approx(
            2.0 * 10 + -0.5 * 5
        )

    def test_id_format_sets_B_via_exchanges_key(self):
        builder = self._builder()
        builder.add({builder.Y[0]: 10, builder.Y[1]: 5})
        model = builder.build(
            {
                "Dirty": {"produces": {"out": 1.0}, "exchanges": {"CO2": 2.0}},
                "Clean": {"produces": {"out": 1.0}, "exchanges": {"CO2": -0.5}},
            }
        )
        assert model.eval(model.structure.ElementaryBalance[0]) == pytest.approx(
            2.0 * 10 + -0.5 * 5
        )

    def test_unknown_exchange_id_in_recipe_raises(self):
        builder = self._builder()
        with pytest.raises(ValueError):
            builder.build(
                {
                    "Dirty": {"produces": {"out": 1.0}, "exchanges": {"N2O": 1.0}},
                }
            )

    def test_get_recipe_includes_exchanges(self):
        builder = self._builder()
        model = builder.build(
            {
                "Dirty": {"produces": {"out": 1.0}, "exchanges": {"CO2": 2.0}},
            }
        )
        recipe = model.get_recipe("Dirty")
        assert recipe["exchanges"]["CO2"] == 2.0

    def test_get_recipe_default_includes_empty_exchanges(self):
        builder = self._builder()
        model = builder.build({"Dirty": {"produces": {"out": 1.0}}})
        recipe = model.get_recipe("Clean")
        assert recipe["exchanges"] == {}

    def test_negative_B_values_preserved_no_clipping(self):
        """Biogenic uptake is a negative B entry -- no nonnegativity assumption."""
        builder = self._builder()
        builder.add({builder.Y[0]: 10})
        model = builder.build(
            {
                "Dirty": {"produces": {"out": 1.0}, "exchanges": {"CO2": -3.5}},
            }
        )
        assert model.eval(model.structure.ElementaryBalance[0]) == pytest.approx(-35.0)


class TestElementaryFlowTable:
    """The structural elementary-flow table lives on ModelStructure; resolve
    it through the model with reporting.evaluate_views."""

    def test_tidy_table(self):
        from flowprog.reporting import evaluate_views

        processes = [
            Process("Dirty", produces=["out"], consumes=[], exchanges=["CO2"]),
            Process("Clean", produces=["out"], consumes=[], exchanges=["CO2"]),
        ]
        objects = [MObject("out", has_market=True)]
        exchanges = [MExchange("CO2")]
        builder = ModelBuilder(processes, objects, exchanges)
        builder.add({builder.Y[0]: 10, builder.Y[1]: 5})
        model = builder.build(
            {
                "Dirty": {"produces": {"out": 1.0}, "exchanges": {"CO2": 2.0}},
                "Clean": {"produces": {"out": 1.0}, "exchanges": {"CO2": -0.5}},
            }
        )

        table = evaluate_views(model, model.structure.elementary_flow_table())
        rows = {
            (row.exchange, row.process): row.value for row in table.itertuples()
        }
        assert rows[("CO2", "Dirty")] == pytest.approx(20.0)
        assert rows[("CO2", "Clean")] == pytest.approx(-2.5)

    def test_empty_when_no_exchanges_declared(self):
        processes = [Process("P1", produces=["out"], consumes=[])]
        objects = [MObject("out")]
        builder = ModelBuilder(processes, objects, [MExchange("CO2")])
        builder.add({builder.Y[0]: 10})
        model = builder.build({"P1": {"produces": {"out": 1.0}}})

        assert len(model.structure.elementary_flow_table()) == 0


class TestLambdifyElementaryTotals:
    def test_lambdify_expressions_with_elementary_balance(self):
        processes = [
            Process("Dirty", produces=["out"], consumes=[], exchanges=["CO2"]),
        ]
        objects = [MObject("out", has_market=True)]
        exchanges = [MExchange("CO2")]
        builder = ModelBuilder(processes, objects, exchanges)
        demand = sy.Symbol("demand", nonnegative=True)
        builder.add(builder.pull_production("out", demand))
        model = builder.build(
            {
                "Dirty": {"produces": {"out": 0.6}, "exchanges": {"CO2": 2.0}},
            }
        )

        # lambdify() resolves structural symbols itself, so the structural
        # expression can be compiled directly -- no eval() pass needed first.
        func = model.lambdify(
            expressions={"total_co2": model.structure.ElementaryBalance[0]}
        )
        result = func({"demand": 100})
        # pull_production("out", demand) sets Y[0] = demand / S[out, Dirty]
        assert result["total_co2"] == pytest.approx(100.0 / 0.6 * 2.0)


class TestSerialisation:
    def test_builder_roundtrip_preserves_exchanges_and_B(self, tmp_path):
        processes = [Process("Dirty", produces=["out"], consumes=[])]
        objects = [MObject("out", has_market=True)]
        exchanges = [MExchange("CO2")]
        builder = ModelBuilder(processes, objects, exchanges)
        builder.add(builder.pull_production("out", sy.Symbol("demand")))

        filepath = tmp_path / "test.json"
        builder.save(str(filepath))
        loaded = ModelBuilder.load(str(filepath))

        assert loaded.structure.elementary_exchanges == tuple(exchanges)
        assert loaded.structure.lookup_exchange("CO2") == 0
        assert loaded.structure.B.shape == builder.structure.B.shape

    def test_model_roundtrip_preserves_B_recipe(self, tmp_path):
        processes = [
            Process("Dirty", produces=["out"], consumes=[], exchanges=["CO2"])
        ]
        objects = [MObject("out", has_market=True)]
        exchanges = [MExchange("CO2")]
        builder = ModelBuilder(processes, objects, exchanges)
        builder.add({builder.Y[0]: 10})
        model = builder.build(
            {"Dirty": {"produces": {"out": 1.0}, "exchanges": {"CO2": -3.5}}}
        )

        filepath = tmp_path / "test.json"
        model.save(str(filepath))
        from flowprog import SympyModel

        loaded = SympyModel.load(str(filepath))

        assert loaded.get_recipe("Dirty")["exchanges"]["CO2"] == -3.5
        assert loaded.eval(loaded.structure.ElementaryBalance[0]) == pytest.approx(-35.0)
