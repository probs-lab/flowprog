import pytest
from rdflib import URIRef

from flowprog.imperative_model import *


# Shorthand
MASS = URIRef("http://qudt.org/vocab/quantitykind/Mass")


def MObject(id, *args, **kwargs):
    return Object(id, MASS, *args, **kwargs)


a = sy.Symbol("a")
b = sy.Symbol("b")


class TestSimpleChain:
    @pytest.fixture
    def m(self):
        processes = [Process("M1", produces=["out"], consumes=["in"])]
        objects = [MObject("in"), MObject("out")]
        return Model(processes, objects)

    def test_push_first_object_consumption(self, m):
        result = m.push_consumption("in", a)
        assert m.eval_intermediates(result[m.X[0]]) == a / m.U[0, 0]
        assert m.eval_intermediates(result[m.Y[0]]) == a / m.U[0, 0]

    def test_push_process_input(self, m):
        result = m.push_process_input("M1", "in", a)
        assert m.eval_intermediates(result[m.X[0]]) == a / m.U[0, 0]
        assert m.eval_intermediates(result[m.Y[0]]) == a / m.U[0, 0]

    def test_pull_process_output(self, m):
        result = m.pull_process_output("M1", "out", a)
        assert m.eval_intermediates(result[m.X[0]]) == a / m.S[1, 0]
        assert m.eval_intermediates(result[m.Y[0]]) == a / m.S[1, 0]

    def test_pull_last_object_production(self, m):
        result = m.pull_production("out", a)
        assert m.eval_intermediates(result[m.X[0]]) == a / m.S[1, 0]
        assert m.eval_intermediates(result[m.Y[0]]) == a / m.S[1, 0]

    def test_label(self, m):
        m.add(m.push_consumption("in", a), label="label")
        assert m.get_history(m.X[0]) == ["label"]

    def test_add_initial(self, m):
        m.add({m.X[0]: 3.5})
        assert m.eval(m.X[0]) == 3.5

    def test_add_adds(self, m):
        m.add({m.X[0]: 3.5})
        m.add({m.X[0]: 2.5})
        assert m.eval(m.X[0]) == 6.0

    def test_add_two_at_once(self, m):
        m.add(
            {m.X[0]: 3.5},
            {m.X[0]: 2.5},
        )
        assert m.eval(m.X[0]) == 6.0

    def test_pull_process_output_error_unknown(self, m):
        with pytest.raises(ValueError):
            m.pull_process_output("M1", "does not exist", 3)

    def test_push_process_input_error_unknown(self, m):
        with pytest.raises(ValueError):
            m.push_process_input("M1", "does not exist", 3)

    def test_pull_process_output_error_wrong_object(self, m):
        """Test that pull_process_output raises error when process doesn't produce the object."""
        with pytest.raises(ValueError, match="Process 'M1' does not produce object 'in'"):
            m.pull_process_output("M1", "in", 3)

    def test_push_process_input_error_wrong_object(self, m):
        """Test that push_process_input raises error when process doesn't consume the object."""
        with pytest.raises(ValueError, match="Process 'M1' does not consume object 'out'"):
            m.push_process_input("M1", "out", 3)


class TestTwoProducersAllocateBackwards:
    @pytest.fixture
    def m(self):
        processes = [
            Process("M1", produces=["out"], consumes=["in1"]),
            Process("M2", produces=["out"], consumes=["in2"]),
        ]
        objects = [MObject("in1"), MObject("in2"), MObject("out")]
        return Model(processes, objects)

    @pytest.fixture
    def m2(self):
        # Like m fixture above but with another layer
        processes = [
            Process("M1", produces=["out"], consumes=["in1"]),
            Process("M2", produces=["out"], consumes=["in2"]),
            Process("U1", produces=["in2"], consumes=["primary"]),
            Process("U2", produces=["in2"], consumes=["primary"]),
        ]
        objects = [
            MObject("primary"),
            MObject("in1"),
            MObject("in2", has_market=True),
            MObject("out"),
        ]
        return Model(processes, objects)

    def test_pull_last_object_production(self, m):
        d = sy.Symbol("d")
        a1 = sy.Symbol("a1")
        a2 = sy.Symbol("a2")
        result = m.pull_production(
            "out",
            d,
            allocate_backwards={
                "out": {
                    "M1": a1,
                    "M2": a2,
                }
            },
        )
        assert m.eval_intermediates(result[m.X[0]]) == d * a1 / m.S[2, 0]
        assert m.eval_intermediates(result[m.X[1]]) == d * a2 / m.S[2, 1]

    def test_error_when_missing_allocate_backwards(self, m):
        d = sy.Symbol("d")
        with pytest.raises(ValueError, match="not defined for out"):
            m.pull_production("out", d, allocate_backwards={})

    def test_error_when_allocate_backwards_has_unknown_process(self, m):
        d = sy.Symbol("d")
        with pytest.raises(ValueError, match="Unknown process id Unknown"):
            m.pull_production("out", d, allocate_backwards={"out": {"Unknown": 1}})

    def test_error_when_allocate_backwards_does_not_add_up(self, m):
        d = sy.Symbol("d")
        with pytest.raises(ValueError, match="does not sum to 1"):
            m.pull_production(
                "out", d, allocate_backwards={"out": {"M1": 0.1, "M2": 0.1}}
            )

    def test_allocate_backwards_not_required_when_not_needed(self, m2):
        d = sy.Symbol("d")
        # Here no production from M2 is required, so the allocation of "in2" is
        # irrelevant.
        m2.pull_production("out", d, allocate_backwards={"out": {"M1": 1, "M2": 0}})

        # Here production for M2 is required, so the allocation is needed.
        with pytest.raises(ValueError, match="not defined for in2"):
            m2.pull_production(
                "out", d, allocate_backwards={"out": {"M1": 0.5, "M2": 0.5}}
            )

        # This should be ok
        m2.pull_production(
            "out",
            d,
            allocate_backwards={
                "out": {"M1": 0.5, "M2": 0.5},
                "in2": {"U1": 0.2, "U2": 0.8},
            },
        )


class TestBalanceObject:
    @pytest.fixture
    def m(self):
        processes = [
            Process("M1", consumes=["in1"], produces=["mid"]),
            Process("M2", consumes=["in2"], produces=["mid"]),
            Process("Use", consumes=["mid"], produces=["out"]),
        ]
        objects = [MObject("in1"), MObject("in2"), MObject("mid"), MObject("out")]
        return Model(processes, objects)

    def test_balance_mid_object(self, m):
        m.add(
            m.pull_production("out", a, until_objects=["mid"]),
            m.push_consumption("in1", b, until_objects=["mid"]),
        )

        assert m.eval(m.X[0]) == b / m.U[0, 0]
        assert m.eval(m.X[1]) == 0
        assert m.eval(m.X[2]) == a / m.S[3, 2]

        # Balance production using process M2
        m.add(m.pull_process_output("M2", "mid", m.object_production_deficit("mid")))

        assert m.eval(m.X[0]) == b / m.U[0, 0]
        assert (
            m.eval(m.X[1])
            == sy.Max(0, a / m.S[3, 2] * m.U[2, 2] - b / m.U[0, 0] * m.S[2, 0])
            / m.S[2, 1]
        )
        assert m.eval(m.X[2]) == a / m.S[3, 2]


class TestLoops:
    @pytest.fixture
    def m(self):
        processes = [
            Process("A", consumes=["B"], produces=["B"]),
        ]
        objects = [MObject("B", True)]
        return Model(processes, objects)

    def test_loop_works(self, m):
        m.pull_production("B", 3)


class TestLoops2:
    @pytest.fixture
    def m(self):
        processes = [
            Process("A", consumes=["1"], produces=["2"]),
            Process("B", consumes=["2"], produces=["1"]),
        ]
        objects = [MObject("1", True), MObject("2", True)]
        return Model(processes, objects)

    def test_loop_works(self, m):
        m.pull_production("1", 3)


class TestExpr:
    @pytest.fixture
    def m(self):
        processes = [
            Process("M1", consumes=["in1"], produces=["mid", "by"]),
            Process("M2", consumes=["in2"], produces=["mid", "by"]),
            Process("Use", consumes=["mid"], produces=["out"]),
        ]
        objects = [
            MObject("in1"),
            MObject("in2"),
            MObject("mid"),
            MObject("by"),
            MObject("out"),
        ]
        return Model(processes, objects)

    def test_expr_process_output(self, m):
        expr = m.expr("ProcessOutput", process_id="M1", object_id="mid")
        assert expr == m.Y[0] * m.S[2, 0]

    def test_expr_process_input(self, m):
        expr = m.expr("ProcessInput", process_id="M1", object_id="in1")
        assert expr == m.X[0] * m.U[0, 0]

    def test_expr_soldproduction(self, m):
        expr = m.expr("SoldProduction", object_id="mid")
        assert expr == (
            m.expr("ProcessOutput", process_id="M1", object_id="mid")
            + m.expr("ProcessOutput", process_id="M2", object_id="mid")
        )

    def test_expr_consumption(self, m):
        expr = m.expr("Consumption", object_id="mid")
        assert expr == (m.expr("ProcessInput", process_id="Use", object_id="mid"))


# def test_solution_longer_chain():
#     # two processes with balancing object in the middle
#     processes = [
#         Process("M1", produces=["mid"], consumes=["in"]),
#         Process("M2", produces=["out"], consumes=["mid"]),
#     ]
#     objects = [Object("in"), Object("mid"), Object("out")]

#     m = Model(processes, objects)
#     independent_vars = list(m.S.values()) + list(m.U.values()) + [m.Z[2]]
#     sol, actual_vars = solve(m, independent_vars)

#     assert set(actual_vars) == set(independent_vars)
#     assert sol == {
#         m.s[2, 1]: m.Z[2],
#         m.Y[   1]: m.Z[2] / m.S[2, 1],
#         m.X[   1]: m.Z[2] / m.S[2, 1],
#         m.u[1, 1]: m.Z[2] / m.S[2, 1] * m.U[1, 1],
#         m.Z[1   ]: m.Z[2] / m.S[2, 1] * m.U[1, 1],
#         m.s[1, 0]: m.Z[2] / m.S[2, 1] * m.U[1, 1],
#         m.Y[   0]: m.Z[2] / m.S[2, 1] * m.U[1, 1] / m.S[1, 0],
#         m.X[   0]: m.Z[2] / m.S[2, 1] * m.U[1, 1] / m.S[1, 0],
#         m.u[0, 0]: m.Z[2] / m.S[2, 1] * m.U[1, 1] / m.S[1, 0] * m.U[0, 0],
#         m.Z[0   ]: m.Z[2] / m.S[2, 1] * m.U[1, 1] / m.S[1, 0] * m.U[0, 0],
#     }


# class TestAlternativeProductionProcessesBackwardsAllocation:

#     @pytest.fixture
#     def m(self):
#         processes = [
#             Process("M1", produces=["out"], consumes=["in1"]),
#             Process("M2", produces=["out"], consumes=["in2"]),
#         ]
#         objects = [Object("in1"), Object("in2"), Object("out", allocate_backwards=True)]
#         m = define_symbols(processes, objects)
#         return m

#     def test_object_equations(self, m):
#         assert list(object_alloc_equations(0, m)) == [
#             (m.Z[0], -m.u[0, 0]),
#         ]
#         assert list(object_alloc_equations(1, m)) == [
#             (m.Z[1], -m.u[1, 1]),
#         ]
#         assert list(object_alloc_equations(2, m)) == [
#             (m.s[2, 0], -m.Z[2] * m.alpha[2, 0]),
#             (m.s[2, 1], -m.Z[2] * (1 - m.alpha[2, 0])),
#         ]

#     def test_solution_output_and_allocation(self, m):
#         # drive from demand for "out"
#         independent_vars = list(m.S.values()) + list(m.U.values()) + [m.Z[2], m.alpha[2, 0]]
#         sol, actual_vars = solve(m, independent_vars)

#         assert set(actual_vars) == set(independent_vars)
#         assert sol[m.X[0]] == m.Z[2] / m.S[2, 0] * m.alpha[2, 0]
#         assert sol[m.X[1]].expand() == (m.Z[2] / m.S[2, 1] * (1 - m.alpha[2, 0])).expand()

#     def test_solution_output_and_one_production(self, m):
#         # drive from demand for "out" and one of the production values
#         independent_vars = list(m.S.values()) + list(m.U.values()) + [m.Z[2], m.X[0]]
#         sol, actual_vars = solve(m, independent_vars)

#         assert set(actual_vars) == set(independent_vars)
#         assert sol[m.X[1]].expand() == ((m.Z[2] - m.X[0] * m.S[2, 0]) / m.S[2, 1]).expand()

#     def test_solution_both_productions_nonlinear_error(self, m):
#         # drive from demand through both of the production values
#         independent_vars = list(m.S.values()) + list(m.U.values()) + [m.X[0], m.X[1]]

#         with pytest.raises(NonlinearModelError):
#             solve(m, independent_vars)

#         # assert set(actual_vars) == set(independent_vars)
#         # assert sol[m.Z[2]].expand() == (m.X[0] * m.S[2, 0] + m.X[1] * m.S[2, 0]).expand()


# class TestAlternativeProductionProcessesNoAllocation:

#     @pytest.fixture
#     def m(self):
#         processes = [
#             Process("M1", produces=["out"], consumes=["in1"]),
#             Process("M2", produces=["out"], consumes=["in2"]),
#         ]
#         objects = [Object("in1"), Object("in2"), Object("out")]
#         m = define_symbols(processes, objects)
#         return m

#     def test_solution_both_productions(self, m):
#         # drive from demand through both of the production values
#         independent_vars = list(m.S.values()) + list(m.U.values()) + [m.X[0], m.X[1]]

#         sol, actual_vars = solve(m, independent_vars)

#         assert set(actual_vars) == set(independent_vars)
#         assert sol[m.Z[2]].expand() == (m.X[0] * m.S[2, 0] + m.X[1] * m.S[2, 1]).expand()

#     def test_solution_output_and_one_production(self, m):
#         # drive from demand for "out" and one of the production values
#         independent_vars = list(m.S.values()) + list(m.U.values()) + [m.Z[2], m.X[0]]
#         sol, actual_vars = solve(m, independent_vars)

#         assert set(actual_vars) == set(independent_vars)
#         assert sol[m.X[1]].expand() == ((m.Z[2] - m.X[0] * m.S[2, 0]) / m.S[2, 1]).expand()

#     def test_no_allocation_variables_created(self, m):
#         assert m.alpha == {}


# # def test_solution_underdetermined():
# #     processes = [Process("M1", produces=["out"], consumes=["in"])]
# #     objects = [Object("in"), Object("out")]

# #     m = define_symbols(processes, objects)
# #     independent_vars = list(m.S.values()) + list(m.U.values()) + [m.Y[1]]
# #     sol, actual_vars = solve(m, independent_vars)
# #     assert sol == {
# #         m.s[1, 0]: m.Y[1],
# #         m.u[0, 0]: m.Y[1] * m.U[0, 0] / m.S[1, 0],
# #     }

# # def test_solution():
# #     processes = [
# #         Process("M1", produces=["P1"], consumes=["PP"]),
# #         Process("M1", produces=["P1"], consumes=["PP", "PE"]),
# #     ]

# #     objects = [
# #         Object("P1", stock=True),
# #         Object("P2", stock=True),
# #         Object("PP"),
# #         Object("PE"),
# #     ]

# #     m = define_symbols(processes, objects)
# #     independent_vars = [m.Delta[0], m.Delta[1]] + list(m.S.values()) + list(m.U.values()) + [m.alpha[0, 0]]
# #     sol = solve(m, independent_vars)
# #     assert sol == 0

# # @pytest.mark.paramtrize("v")
# # flows = solution_to_flows(m, sol, {m.Delta[0]: Delta0, m.Delta[1]: Delta1, m.S[0, 0]: 1, m.S[0, 1]: 1, m.U[2, 0]: 1, m.U[2, 1]: frac, m.U[3, 1]: 1-frac,
# #                                        m.alpha[0, 0]: alpha00})
# #     w.links = flows.to_dict(orient='records')
