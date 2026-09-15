"""Property: a single-step pull_production() matches a Leontief/LCA solve.

For a model where no process produces more than one object, has stock, or
sits on a cycle through a market object, one call to
``builder.pull_production(object_id, demand, ...)`` computes the same
process activity levels as directly solving the linear technosphere system
implied by the same recipe data: one activity level x[j] per process and
one market-cleared quantity d[i] per object, related by

    S[i,j]*x[j] = alpha[i,j]*d[i]                               for each (object i, producer j)
    d[i] = external[i] + sum_{k consuming i} U[i,k]*x[k]        if i has a market and isn't cut off
    d[i] = external[i]                                          otherwise

where `external[i]` is the pulled `demand` if `i == object_id` else 0, and
`alpha[i,j]` is `allocate_backwards[i][j]` (or 1 if `i` has only one
producer). `until_objects` cuts recursion off at particular objects,
treating them as boundary inputs instead of resolving their own
production; the matrix equivalent is simply not writing a market equation
for them, same as an object with no market.

Three restrictions on the generated models, each demonstrated below:

- No process may have ``has_stock=True``: `pull_production` only mirrors
  an activity's output onto its input (X[j] = Y[j]) for stock-free
  processes.

- No process may produce more than one object. A co-producing process running at
  one activity level x makes every one of its outputs at once (in the ratio
  fixed by S), but `pull_production` does not enforce that: it computes each
  output's activity independently and adds them, so there is generally no single
  x consistent with the linear system above. (This behaviour might change in
  future)

- No cycle through an object with ``has_market=True``: `pull_production`'s loop
  guard unrolls a cycle exactly once and stops rather than iterating to a fixed
  point, so its result need not match the linear solve even when that solve is
  well-posed (a convergent cycle) and always fails to match it when it isn't (a
  singular one). (This behaviour might change in future)

"""

import numpy as np
import pytest
from hypothesis import given, settings, assume, example, strategies as st
from rdflib import URIRef

from flowprog import ModelBuilder, ModelStructure, Process, Object
from .model_strategies import model_builder_strategy

MASS = URIRef("http://qudt.org/vocab/quantitykind/Mass")


def _alpha(structure, allocate_backwards, object_id, process_id):
    """Share of `object_id`'s demand routed to `process_id` (1.0 if sole producer)."""
    producers = structure.producers_of(object_id)
    if len(producers) == 1:
        return 1.0
    return allocate_backwards[object_id].get(process_id, 0.0)


def leontief_solve(structure, object_id, demand, s_vals, u_vals, allocate_backwards, until_objects):
    """Solve the linear technosphere system described in the module docstring.

    Unknowns are `x[j]` (process activity) and `d[i]` (object demand), `M +
    N` in total, solved as one `(M+N) x (M+N)` linear system; `x` is
    returned. Requires every process to produce at most one object.
    """
    processes = structure.processes
    objects = structure.objects
    M = len(processes)
    N = len(objects)
    stop_objects = set(until_objects) | {object_id}

    def x_col(j):
        return j

    def d_col(i):
        return M + i

    coeffs = np.zeros((M + N, M + N))
    rhs = np.zeros(M + N)

    # x[j] - alpha[i,j]/S[i,j] * d[i] = 0, for j's sole produced object i (if any)
    for j, proc in enumerate(processes):
        coeffs[x_col(j), x_col(j)] = 1.0
        for obj_id in proc.produces:
            i = structure.lookup_object(obj_id)
            alpha = _alpha(structure, allocate_backwards, obj_id, proc.id)
            coeffs[x_col(j), d_col(i)] -= alpha / s_vals[(proc.id, obj_id)]

    # d[i] - sum_k U[i,k]*x[k] = external[i], for objects with a market that
    # propagates; d[i] = external[i] otherwise.
    for obj in objects:
        i = structure.lookup_object(obj.id)
        coeffs[d_col(i), d_col(i)] = 1.0
        rhs[d_col(i)] = demand if obj.id == object_id else 0.0
        if obj.has_market and obj.id not in stop_objects:
            for proc in processes:
                if obj.id in proc.consumes:
                    k = structure.lookup_process(proc.id)
                    coeffs[d_col(i), x_col(k)] -= u_vals[(proc.id, obj.id)]

    return np.linalg.solve(coeffs, rhs)[:M]


def _recipe_from_values(structure, s_vals, u_vals):
    return {
        p.id: {
            "consumes": {obj: u_vals[(p.id, obj)] for obj in p.consumes},
            "produces": {obj: s_vals[(p.id, obj)] for obj in p.produces},
        }
        for p in structure.processes
    }


# ============================================================================
# Hypothesis strategy
# ============================================================================


_RECIPE_VALUE = st.floats(min_value=0.2, max_value=5.0, allow_nan=False, allow_infinity=False)
# Occasionally exactly 0.0, to exercise pull_production's `if alpha.is_zero:
# continue` (a producer dropped from allocate_backwards entirely).
_WEIGHT = st.one_of(
    st.just(0.0), st.floats(min_value=0.05, max_value=1.0, allow_nan=False, allow_infinity=False)
)
_DEMAND = st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)


@st.composite
def linear_leontief_case(draw, max_processes=4, max_objects=5):
    builder = draw(
        model_builder_strategy(
            min_processes=1,
            max_processes=max_processes,
            min_objects=1,
            max_objects=max_objects,
        )
    )
    structure = builder.structure
    # has_stock=False is model_builder_strategy's default. No co-production:
    assume(all(len(p.produces) <= 1 for p in structure.processes))

    produced_objects = sorted({obj_id for p in structure.processes for obj_id in p.produces})
    assume(produced_objects)
    object_id = draw(st.sampled_from(produced_objects))

    s_vals = {}
    u_vals = {}
    for p in structure.processes:
        for obj in p.produces:
            s_vals[(p.id, obj)] = draw(_RECIPE_VALUE)
        for obj in p.consumes:
            u_vals[(p.id, obj)] = draw(_RECIPE_VALUE)

    allocate_backwards = {}
    for obj in structure.objects:
        producers = structure.producers_of(obj.id)
        if len(producers) > 1:
            weights = draw(
                st.lists(_WEIGHT, min_size=len(producers), max_size=len(producers))
            )
            total = sum(weights)
            assume(total > 0)
            allocate_backwards[obj.id] = {
                pid: w / total for pid, w in zip(producers, weights)
            }

    # Candidate objects to cut off with until_objects: any market object other
    # than the root (cutting the root is already implicit).
    market_objects = [o.id for o in structure.objects if o.has_market and o.id != object_id]
    if market_objects:
        until_objects = draw(st.sets(st.sampled_from(market_objects)))
    else:
        until_objects = set()

    demand = draw(_DEMAND)

    return builder, object_id, demand, s_vals, u_vals, allocate_backwards, until_objects


# ============================================================================
# The property
# ============================================================================


@given(linear_leontief_case())
# Two producers (P0, P1) of a multi-producer object R that both consume the
# same further-upstream, single-producer object M. Satisfying both branches
# of R's demand means computing PM's activity twice (independently, once
# per branch) and summing the two -- the case `merge_activities` exists to
# handle, and one random generation rarely reaches on its own.
@example(
    case=(
        ModelBuilder(
            [
                Process("P0", consumes=["M"], produces=["R"]),
                Process("P1", consumes=["M"], produces=["R"]),
                Process("PM", consumes=[], produces=["M"]),
            ],
            [
                Object("R", MASS, has_market=False),
                Object("M", MASS, has_market=True),
            ],
        ),
        "R",
        10.0,
        {("P0", "R"): 1.0, ("P1", "R"): 1.0, ("PM", "M"): 1.0},
        {("P0", "M"): 1.0, ("P1", "M"): 1.0},
        {"R": {"P0": 0.5, "P1": 0.5}},
        set(),
    )
)
@settings(deadline=None, max_examples=300)
def test_single_step_pull_production_matches_leontief_solve(case):
    builder, object_id, demand, s_vals, u_vals, allocate_backwards, until_objects = case
    structure = builder.structure

    activity = builder.pull_production(
        object_id,
        demand,
        until_objects=until_objects,
        allocate_backwards=allocate_backwards,
    )
    builder.add(activity)

    recipe_data = _recipe_from_values(structure, s_vals, u_vals)
    model = builder.build(recipe_data)

    x_expected = leontief_solve(
        structure, object_id, demand, s_vals, u_vals, allocate_backwards, until_objects
    )

    for j, proc in enumerate(structure.processes):
        y_actual = float(model.eval(model.Y[j]))
        x_actual = float(model.eval(model.X[j]))

        assert x_actual == pytest.approx(y_actual, rel=1e-9, abs=1e-9), (
            f"X[{proc.id}] != Y[{proc.id}]"
        )
        assert y_actual == pytest.approx(x_expected[j], rel=1e-6, abs=1e-6), (
            f"process {proc.id}: pull_production gave {y_actual}, "
            f"Leontief solve gave {x_expected[j]}"
        )


# ============================================================================
# The three restrictions, demonstrated
# ============================================================================


def test_coproduction_has_no_consistent_activity_level():
    """PJ co-produces A (S=2) and B (S=3), each pulled to 10 units via
    independent downstream chains. Physically, one activity level for PJ
    can't supply 10 of each: x=5 from A's side, x=10/3 from B's side.
    `pull_production` doesn't resolve this -- it sums the two chains'
    independently-implied activities, Y[PJ] = 10/2 + 10/3 = 8.33, matching
    neither (running PJ that fast actually makes 16.67 of A and 25 of B).
    """
    processes = [
        Process("PJ", consumes=[], produces=["A", "B"]),
        Process("PC1", consumes=["A"], produces=["FinalA"]),
        Process("PC2", consumes=["B"], produces=["FinalB"]),
        Process("PR", consumes=["FinalA", "FinalB"], produces=["Root"]),
    ]
    objects = [
        Object("A", MASS, has_market=True),
        Object("B", MASS, has_market=True),
        Object("FinalA", MASS, has_market=True),
        Object("FinalB", MASS, has_market=True),
        Object("Root", MASS, has_market=False),
    ]
    s_vals = {
        ("PJ", "A"): 2.0,
        ("PJ", "B"): 3.0,
        ("PC1", "FinalA"): 1.0,
        ("PC2", "FinalB"): 1.0,
        ("PR", "Root"): 1.0,
    }
    u_vals = {
        ("PC1", "A"): 1.0,
        ("PC2", "B"): 1.0,
        ("PR", "FinalA"): 1.0,
        ("PR", "FinalB"): 1.0,
    }
    demand = 10.0

    structure = ModelStructure(processes, objects)
    builder = ModelBuilder(processes, objects)
    builder.add(builder.pull_production("Root", demand))
    model = builder.build(_recipe_from_values(structure, s_vals, u_vals))
    y_pj = float(model.eval(model.Y[0]))

    implied_by_a_alone = demand / s_vals[("PJ", "A")]
    implied_by_b_alone = demand / s_vals[("PJ", "B")]
    assert y_pj == pytest.approx(implied_by_a_alone + implied_by_b_alone)
    assert s_vals[("PJ", "A")] * y_pj != pytest.approx(demand)
    assert s_vals[("PJ", "B")] * y_pj != pytest.approx(demand)


def test_cycle_through_market_object_makes_the_matrix_singular():
    """R is pulled from P0, which needs both X and Y; X and Z feed each other
    in a cycle that doesn't pass through R. `pull_production` still returns
    some finite numbers (its loop guard always terminates), but the
    technosphere matrix is exactly singular: no unique solution exists for
    `pull_production`'s result to match.
    """
    processes = [
        Process("P0", consumes=["X", "Y"], produces=["R"]),
        Process("P1", consumes=["Z"], produces=["X"]),
        Process("P2", consumes=["Z"], produces=["Y"]),
        Process("P3", consumes=["X"], produces=["Z"]),
    ]
    objects = [
        Object("R", MASS, has_market=False),
        Object("X", MASS, has_market=True),
        Object("Y", MASS, has_market=True),
        Object("Z", MASS, has_market=True),
    ]
    structure = ModelStructure(processes, objects)
    s_vals = {("P0", "R"): 1.0, ("P1", "X"): 1.0, ("P2", "Y"): 1.0, ("P3", "Z"): 1.0}
    u_vals = {
        ("P0", "X"): 1.0,
        ("P0", "Y"): 1.0,
        ("P1", "Z"): 1.0,
        ("P2", "Z"): 1.0,
        ("P3", "X"): 1.0,
    }

    with pytest.raises(np.linalg.LinAlgError):
        leontief_solve(structure, "R", 1.0, s_vals, u_vals, {}, set())


def test_convergent_loop_is_well_posed_but_pull_production_is_still_wrong():
    """Not every market cycle is singular. Here R needs X; X and Z feed each
    other in a loop (X needs Z, Z needs X) with combined loop gain g1*g2 =
    0.25 -- comfortably convergent, the shape of an ordinary partial-
    recycling loop. The linear system has a unique, finite, easily
    hand-checked solution (a geometric series, `demand / (1 - g1*g2)`), but
    `pull_production` only unrolls the loop once, so it comes up short.
    """
    g1, g2 = 0.5, 0.5
    processes = [
        Process("P0", consumes=["X"], produces=["R"]),
        Process("P1", consumes=["Z"], produces=["X"]),
        Process("P2", consumes=["X"], produces=["Z"]),
    ]
    objects = [
        Object("R", MASS, has_market=False),
        Object("X", MASS, has_market=True),
        Object("Z", MASS, has_market=True),
    ]
    recipe_data = {
        "P0": {"consumes": {"X": 1.0}, "produces": {"R": 1.0}},
        "P1": {"consumes": {"Z": g1}, "produces": {"X": 1.0}},
        "P2": {"consumes": {"X": g2}, "produces": {"Z": 1.0}},
    }
    s_vals = {("P0", "R"): 1.0, ("P1", "X"): 1.0, ("P2", "Z"): 1.0}
    u_vals = {("P0", "X"): 1.0, ("P1", "Z"): g1, ("P2", "X"): g2}
    demand = 10.0

    structure = ModelStructure(processes, objects)
    x_expected = leontief_solve(structure, "R", demand, s_vals, u_vals, {}, set())
    expected_x_p1 = demand / (1 - g1 * g2)
    assert x_expected[1] == pytest.approx(expected_x_p1)

    builder = ModelBuilder(processes, objects)
    builder.add(builder.pull_production("R", demand))
    model = builder.build(recipe_data)
    y_actual_p1 = float(model.eval(model.Y[1]))

    assert y_actual_p1 == pytest.approx(demand)
    assert y_actual_p1 != pytest.approx(expected_x_p1)
