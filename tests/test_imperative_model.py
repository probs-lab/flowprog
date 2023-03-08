#!/usr/bin/env python3

import pytest

from mass_flow_model.imperative_model import *


class TestSimpleChain:

    @pytest.fixture
    def builder(self):
        processes = [Process("M1", produces=["out"], consumes=["in"])]
        objects = [Object("in"), Object("out")]
        m = define_symbols(processes, objects)
        return ModelBuilder(m)

    def test_push_first_object_consumption(self, builder):
        a = sy.Symbol("a")
        builder.push_consumption("in", a)
        m = builder.m
        assert m.X[0].value == a / m.U[0, 0]
        assert m.X[0].history == ['set input of in into M1 = a']
        assert m.X[0].value == m.Y[0].value

    def test_push_process_input(self, builder):
        a = sy.Symbol("a")
        builder.push_process_input("M1", "in", a)
        m = builder.m
        assert m.X[0].value == a / m.U[0, 0]
        assert m.X[0].history == ['set input of in into M1 = a']
        assert m.X[0].value == m.Y[0].value

    def test_pull_process_output(self, builder):
        a = sy.Symbol("a")
        builder.pull_process_output("M1", "out", a)
        m = builder.m
        assert m.X[0].value == a / m.S[1, 0]
        assert m.X[0].history == ['set output of out from M1 = a']
        assert m.X[0].value == m.Y[0].value

    def test_pull_last_object_production(self, builder):
        a = sy.Symbol("a")
        builder.pull_production("out", a)
        m = builder.m
        assert m.X[0].value == a / m.S[1, 0]
        assert m.X[0].history == ['set output of out from M1 = a']
        assert m.X[0].value == m.Y[0].value

    def test_pull_process_output_error_unknown(self, builder):
        with pytest.raises(ValueError):
            builder.pull_process_output("M1", "does not exist", 3)

    def test_push_process_input_error_unknown(self, builder):
        with pytest.raises(ValueError):
            builder.push_process_input("M1", "does not exist", 3)


class TestTwoProducersAllocateBackwards:

    @pytest.fixture
    def builder(self):
        processes = [
            Process("M1", produces=["out"], consumes=["in1"]),
            Process("M2", produces=["out"], consumes=["in2"]),
        ]
        objects = [Object("in1"), Object("in2"), Object("out")]
        m = define_symbols(processes, objects)
        return ModelBuilder(m)

    def test_pull_last_object_production(self, builder):
        d = sy.Symbol("d")
        a1 = sy.Symbol("a1")
        a2 = sy.Symbol("a2")
        builder.pull_production("out", d, allocate_backwards={
            "out": {
                "M1": a1,
                "M2": a2,
            }
        })
        m = builder.m
        assert m.X[0].value == d * a1 / m.S[2, 0]
        assert m.X[1].value == d * a2 / m.S[2, 1]
        # assert m.X == m.Y


class TestBalanceObject:

    @pytest.fixture
    def builder(self):
        processes = [
            Process("M1", consumes=["in1"], produces=["mid"]),
            Process("M2", consumes=["in2"], produces=["mid"]),
            Process("Use", consumes=["mid"], produces=["out"]),
        ]
        objects = [Object("in1"), Object("in2"), Object("mid"), Object("out")]
        m = define_symbols(processes, objects)
        return ModelBuilder(m)

    def test_balance_mid_object(self, builder):
        a = sy.Symbol("a")
        b = sy.Symbol("b")
        m = builder.m

        builder.pull_production("out", a, until_objects=["mid"])
        builder.push_consumption("in1", b, until_objects=["mid"])

        assert m.X[0].value == b / m.U[0, 0]
        assert m.X[1].value == None
        assert m.X[2].value == a / m.S[3, 2]

        builder.balance_production("mid", "M2")

        assert m.X[0].value == b / m.U[0, 0]
        assert m.X[1].value == sy.Max(0, a / m.S[3, 2] * m.U[2, 2] - b / m.U[0, 0] * m.S[2, 0]) / m.S[2, 1]
        assert m.X[2].value == a / m.S[3, 2]


class TestLoops:

    @pytest.fixture
    def builder(self):
        processes = [
            Process("A", consumes=["B"], produces=["B"]),
        ]
        objects = [Object("B")]
        m = define_symbols(processes, objects)
        return ModelBuilder(m)

    @pytest.mark.skip
    def test_loop_works(self, builder):
        builder.pull_production("B", 3)


# def test_solution_longer_chain():
#     # two processes with balancing object in the middle
#     processes = [
#         Process("M1", produces=["mid"], consumes=["in"]),
#         Process("M2", produces=["out"], consumes=["mid"]),
#     ]
#     objects = [Object("in"), Object("mid"), Object("out")]

#     m = define_symbols(processes, objects)
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
