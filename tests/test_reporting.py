"""Tests for flowprog.reporting: groupings, characterisation, aggregation."""

import logging

import pandas as pd
import pytest
import sympy as sy
from rdflib import URIRef

from flowprog import ModelBuilder, Process, Object, ElementaryExchange
from flowprog.reporting import Reporting, Grouping

MASS = URIRef("http://qudt.org/vocab/quantitykind/Mass")


def MObject(id, *args, **kwargs):
    return Object(id, MASS, *args, **kwargs)


def MExchange(id, *args, **kwargs):
    return ElementaryExchange(id, MASS, *args, **kwargs)


def build_model():
    processes = [
        Process("Dirty", produces=["out"], consumes=[]),
        Process("Clean", produces=["out"], consumes=[]),
        Process("Use", produces=[], consumes=["out"]),
    ]
    objects = [MObject("out", has_market=True)]
    exchanges = [MExchange("CO2"), MExchange("CH4")]
    builder = ModelBuilder(processes, objects, exchanges)

    demand = sy.Symbol("demand", positive=True)
    builder.add(builder.push_consumption("out", demand))
    builder.add(builder.pull_process_output("Dirty", "out", demand * sy.Rational(3, 4)))
    builder.add(builder.pull_process_output("Clean", "out", demand * sy.Rational(1, 4)))

    recipe = {
        "Dirty": {"produces": {"out": 1.0}, "exchanges": {"CO2": 2.0, "CH4": 0.1}},
        "Clean": {"produces": {"out": 1.0}, "exchanges": {"CO2": 0.5}},
        "Use": {"consumes": {"out": 1.0}},
    }
    model = builder.build(recipe)
    return model, demand


GWP = {"CO2": 1, "CH4": 28}


class TestGrouping:
    def test_disjoint_groups_build_ok(self):
        mapping = Grouping.build(
            "stage",
            {"upstream": {"Dirty", "Clean"}, "downstream": {"Use"}},
            all_ids={"Dirty", "Clean", "Use"},
        )
        assert mapping == {"Dirty": "upstream", "Clean": "upstream", "Use": "downstream"}

    def test_overlap_raises(self):
        with pytest.raises(ValueError):
            Grouping.build(
                "stage",
                {"a": {"Dirty", "Clean"}, "b": {"Clean"}},
                all_ids={"Dirty", "Clean"},
            )

    def test_unknown_id_in_group_raises(self):
        with pytest.raises(ValueError):
            Grouping.build(
                "stage", {"a": {"DoesNotExist"}}, all_ids={"Dirty", "Clean"}
            )

    def test_remainder_assigned_to_other_and_warns(self, caplog):
        with caplog.at_level(logging.WARNING):
            mapping = Grouping.build(
                "stage", {"upstream": {"Dirty"}}, all_ids={"Dirty", "Clean"}
            )
        assert mapping["Clean"] == "other"
        assert any("other" in rec.message for rec in caplog.records)

    def test_no_warning_when_fully_covered(self, caplog):
        with caplog.at_level(logging.WARNING):
            Grouping.build(
                "stage",
                {"upstream": {"Dirty"}, "downstream": {"Clean"}},
                all_ids={"Dirty", "Clean"},
            )
        assert len(caplog.records) == 0


class TestReportingAxisInference:
    def test_process_axis_grouping(self):
        model, demand = build_model()
        rep = Reporting(
            model,
            groupings={"stage": {"upstream": {"Dirty", "Clean"}, "downstream": {"Use"}}},
        )
        assert rep._groupings["stage"][0] == "process"

    def test_exchange_axis_grouping(self):
        model, demand = build_model()
        rep = Reporting(
            model, groupings={"ghg_kind": {"fossil": {"CO2"}, "other": {"CH4"}}}
        )
        assert rep._groupings["ghg_kind"][0] == "exchange"

    def test_unrecognised_ids_raise(self):
        model, demand = build_model()
        with pytest.raises(ValueError):
            Reporting(model, groupings={"bad": {"x": {"NotAnId"}}})


class TestAggregateScalar:
    def test_uncharacterised_total_symbolic(self):
        model, demand = build_model()
        rep = Reporting(model)
        total = rep.aggregate(None)
        # Dirty: demand*3/4*2.0 (CO2) + demand*3/4*0.1 (CH4); Clean: demand*1/4*0.5 (CO2)
        expected = demand * sy.Rational(3, 4) * 2.0 + demand * sy.Rational(
            3, 4
        ) * 0.1 + demand * sy.Rational(1, 4) * 0.5
        # Compare numerically: the raw-then-resolve aggregation path may
        # reassociate float sums at the 1e-16 level.
        assert float(total.subs(demand, 100)) == pytest.approx(
            float(expected.subs(demand, 100))
        )

    def test_characterised_total_numeric(self):
        model, demand = build_model()
        rep = Reporting(model, characterisations={"GWP": GWP})
        total = rep.aggregate("GWP", values={demand: 100})
        # Dirty: CO2=75*2=150, CH4=75*0.1=7.5; Clean: CO2=25*0.5=12.5
        expected = (150 + 12.5) * 1 + 7.5 * 28
        assert total == pytest.approx(expected)

    def test_missing_characterisation_factor_defaults_zero(self):
        model, demand = build_model()
        # GWP_CO2_only omits CH4 -- CH4 flows should contribute zero
        rep = Reporting(model, characterisations={"GWP_CO2_only": {"CO2": 1}})
        total = rep.aggregate("GWP_CO2_only", values={demand: 100})
        expected = 150 + 12.5
        assert total == pytest.approx(expected)

    def test_unknown_characterisation_raises(self):
        model, demand = build_model()
        rep = Reporting(model)
        with pytest.raises(KeyError):
            rep.aggregate("NoSuchCharacterisation")


class TestAggregateByGrouping:
    def _rep(self):
        model, demand = build_model()
        rep = Reporting(
            model,
            groupings={"stage": {"upstream": {"Dirty", "Clean"}}},
            characterisations={"GWP": GWP},
        )
        return rep, demand

    def test_by_grouping_returns_series(self):
        rep, demand = self._rep()
        result = rep.aggregate("GWP", by="stage", values={demand: 100})
        assert isinstance(result, pd.Series)
        assert result["upstream"] == pytest.approx((150 + 12.5) * 1 + 7.5 * 28)

    def test_by_process_raw_id(self):
        rep, demand = self._rep()
        result = rep.aggregate("GWP", by="process", values={demand: 100})
        assert result["Dirty"] == pytest.approx(150 * 1 + 7.5 * 28)
        assert result["Clean"] == pytest.approx(12.5 * 1)

    def test_by_exchange_raw_id(self):
        rep, demand = self._rep()
        result = rep.aggregate(None, by="exchange", values={demand: 100})
        assert result["CO2"] == pytest.approx(150 + 12.5)
        assert result["CH4"] == pytest.approx(7.5)

    def test_by_two_keys_returns_multiindex_series(self):
        rep, demand = self._rep()
        result = rep.aggregate(None, by=("stage", "exchange"), values={demand: 100})
        assert result[("upstream", "CO2")] == pytest.approx(150 + 12.5)
        assert result[("upstream", "CH4")] == pytest.approx(7.5)

    def test_unknown_grouping_name_raises(self):
        rep, demand = self._rep()
        with pytest.raises(ValueError):
            rep.aggregate("GWP", by="not_a_grouping")


class TestLimitToProcesses:
    def test_limit_to_processes_restricts_rows(self):
        model, demand = build_model()
        rep = Reporting(model, characterisations={"GWP": GWP})
        total = rep.aggregate(
            "GWP", limit_to_processes={"Dirty"}, values={demand: 100}
        )
        assert total == pytest.approx(150 * 1 + 7.5 * 28)


class TestTable:
    def test_table_has_grouping_columns(self):
        model, demand = build_model()
        rep = Reporting(
            model, groupings={"stage": {"upstream": {"Dirty", "Clean"}}}
        )
        table = rep.table()
        assert "stage" in table.columns
        assert set(table["stage"]) <= {"upstream", "other"}

    def test_table_values_evaluated_when_given(self):
        model, demand = build_model()
        rep = Reporting(model)
        table = rep.table(values={demand: 100})
        rows = {(r.exchange, r.process): r.value for r in table.itertuples()}
        assert rows[("CO2", "Dirty")] == pytest.approx(150)


class TestRawPath:
    """The lazy/raw flows path: expressions with intermediates unresolved,
    resolved once via lambdify (or eval) instead of eagerly per row."""

    def test_raw_flows_resolve_to_eager_values(self):
        model, demand = build_model()
        eager = model.to_elementary_flows(values={demand: 100})
        raw = model.to_elementary_flows(raw=True)
        resolved = {
            (r.exchange, r.process): float(model.eval(r.value, {demand: 100}))
            for r in raw.itertuples()
        }
        for r in eager.itertuples():
            assert resolved[(r.exchange, r.process)] == pytest.approx(float(r.value))

    def test_raw_with_values_raises(self):
        model, demand = build_model()
        with pytest.raises(ValueError):
            model.to_elementary_flows(values={demand: 100}, raw=True)

    def test_raw_aggregate_feeds_single_lambdify(self):
        model, demand = build_model()
        rep = Reporting(model, characterisations={"GWP": GWP})
        raw_total = rep.aggregate("GWP", raw=True)
        func = model.lambdify(expressions={"total": raw_total})
        assert func({demand: 100})["total"] == pytest.approx(
            rep.aggregate("GWP", values={demand: 100})
        )

    def test_raw_grouped_aggregate_feeds_single_lambdify(self):
        model, demand = build_model()
        rep = Reporting(model, groupings={"stage": {"upstream": {"Dirty", "Clean"}}})
        raw_series = rep.aggregate(None, by=("stage", "exchange"), raw=True)
        exprs = {k: v for k, v in raw_series.items()}
        func = model.lambdify(expressions=exprs)
        numeric = rep.aggregate(None, by=("stage", "exchange"), values={demand: 100})
        results = func({demand: 100})
        for key, expected in numeric.items():
            assert results[key] == pytest.approx(expected)


class TestProductionConsumption:
    """production()/consumption(): the technosphere analogue of aggregate(),
    a group-by over (process, object) cells instead of (process, exchange)
    cells."""

    def test_production_scalar(self):
        model, demand = build_model()
        rep = Reporting(model)
        total = rep.production("out", values={demand: 100})
        assert total == pytest.approx(100)

    def test_consumption_scalar(self):
        model, demand = build_model()
        rep = Reporting(model)
        total = rep.consumption("out", values={demand: 100})
        assert total == pytest.approx(100)

    def test_production_by_process(self):
        model, demand = build_model()
        rep = Reporting(model)
        result = rep.production("out", by="process", values={demand: 100})
        assert isinstance(result, pd.Series)
        assert result["Dirty"] == pytest.approx(75)
        assert result["Clean"] == pytest.approx(25)

    def test_consumption_by_declared_grouping(self):
        model, demand = build_model()
        rep = Reporting(
            model,
            groupings={"stage": {"upstream": {"Dirty", "Clean"}, "downstream": {"Use"}}},
        )
        result = rep.consumption("out", by="stage", values={demand: 100})
        assert result["downstream"] == pytest.approx(100)

    def test_production_by_stage_symbolic(self):
        model, demand = build_model()
        rep = Reporting(
            model, groupings={"stage": {"upstream": {"Dirty", "Clean"}}}
        )
        result = rep.production("out", by="stage")
        assert float(result["upstream"].subs(demand, 100)) == pytest.approx(100)

    def test_limit_to_processes(self):
        model, demand = build_model()
        rep = Reporting(model)
        total = rep.production(
            "out", limit_to_processes={"Dirty"}, values={demand: 100}
        )
        assert total == pytest.approx(75)

    def test_raw_feeds_single_lambdify(self):
        model, demand = build_model()
        rep = Reporting(model)
        raw_total = rep.consumption("out", raw=True)
        func = model.lambdify(expressions={"total": raw_total})
        assert func({demand: 100})["total"] == pytest.approx(
            rep.consumption("out", values={demand: 100})
        )

    def test_grouping_by_exchange_axis_rejected(self):
        model, demand = build_model()
        rep = Reporting(
            model, groupings={"ghg_kind": {"fossil": {"CO2"}, "other": {"CH4"}}}
        )
        with pytest.raises(ValueError):
            rep.production("out", by="ghg_kind")


class TestNamedTotals:
    def test_define_and_evaluate_total(self):
        model, demand = build_model()
        rep = Reporting(
            model,
            groupings={"stage": {"upstream": {"Dirty", "Clean"}}},
            characterisations={"GWP": GWP},
        )
        rep.define_total("GWP_total", characterisation="GWP")
        rep.define_total("GWP_by_stage", characterisation="GWP", by="stage")

        results = rep.totals(values={demand: 100})
        assert results["GWP_total"] == pytest.approx(rep.aggregate("GWP", values={demand: 100}))
        pd.testing.assert_series_equal(
            results["GWP_by_stage"].sort_index(),
            rep.aggregate("GWP", by="stage", values={demand: 100}).sort_index(),
        )

    def test_total_single_lookup(self):
        model, demand = build_model()
        rep = Reporting(model, characterisations={"GWP": GWP})
        rep.define_total("GWP_total", characterisation="GWP")
        assert rep.total("GWP_total", values={demand: 100}) == pytest.approx(
            rep.aggregate("GWP", values={demand: 100})
        )
