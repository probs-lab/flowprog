import pytest
from hypothesis import strategies as st, given, assume, example
from sympy.abc import a, b, c
from rdflib import URIRef
from flowprog.imperative_model import *


# Shorthand
MASS = URIRef('http://qudt.org/vocab/quantitykind/Mass')
def MObject(id, *args, **kwargs):
    return Object(id, MASS, *args, **kwargs)


@given(st.floats(min_value=0, allow_infinity=False),
       st.floats(min_value=0, allow_infinity=False),
       st.floats(min_value=0))
def test_limit_with_symbols(initial, consumption, limit_value):
    processes = [
        Process("P1", consumes=["in"], produces=["mid"]),
        Process("P2", consumes=["mid"], produces=["out"]),
    ]
    objects = [MObject("in"), MObject("mid", True), MObject("out")]
    m = Model(processes, objects)

    sy.var("a, b, c", real=True)

    # First flows
    m.add(m.pull_production("out", a))

    # Add extra demand with limit
    unlimited_extra = m.pull_production("out", b)
    limited_extra = m.limit(unlimited_extra, m.Y[0], c)
    m.add(limited_extra)

    # We should not have exceeded the limit
    value = m.eval(m.Y[0]).subs({
        a: initial,
        b: consumption,
        c: limit_value,
        m.U[0, 0]: 1.0,
        m.S[1, 0]: 0.6,
        m.U[1, 1]: 2.2,
        m.S[2, 1]: 2.0,
    })
    assert value >= initial
    # exceeded = max(0, value - max(initial, limit_value))
    # assert exceeded <= 1e-6
    assert value <= max(initial / 2.0 * 2.2 / 0.6 * 1.00001 + 1e-3, limit_value)


# TODO: check that there is a test that fails if you swap `v` to `proposed` in
# the implementation of `limit` (middle case)

@given(st.floats(min_value=0, allow_infinity=False),
       st.floats(min_value=0, allow_infinity=False),
       st.floats(min_value=0, allow_infinity=False))
@example(2, 3, 10)
def test_limit_can_be_arbitrary_expression(initial1, initial2, consumption):
    processes = [
        Process("P1", consumes=["in"], produces=["mid"]),
        Process("P2", consumes=["mid"], produces=["out1"]),
        Process("P3", consumes=["mid"], produces=["out2"]),
    ]
    objects = [MObject("in"), MObject("mid", True), MObject("out1"), MObject("out2")]
    m = Model(processes, objects)

    sy.var("a, b, c", real=True)

    # First flows
    m.add(
        m.pull_production("out1", a, until_objects={"mid"}),
        m.pull_production("out2", b, until_objects={"mid"}),
    )

    # Add extra demand with limit
    unlimited_extra = m.pull_process_output("P1", "mid", c)
    limit_expr = m.X[1] * m.U[1, 1] + m.X[2] * m.U[1, 2]
    limited_extra = m.limit(
        unlimited_extra,
        m.Y[0] * m.S[1, 0],
        limit_expr,
    )
    m.add(limited_extra)

    # We should not have exceeded the limit
    data = {
        a: initial1,
        b: initial2,
        c: consumption,
        m.U[0, 0]: 1,
        m.S[1, 0]: 1,
        m.U[1, 1]: 1,
        m.S[2, 1]: 1,
        m.U[1, 2]: 1,
        m.S[3, 2]: 1,
    }
    value = m.eval(m.Y[0]).subs(data)
    assert value >= min(initial1 + initial2, consumption)
    assert value <= initial1 + initial2


# Define arbitrary models for testing

@st.composite
def model_strategy(draw):
    num_processes = draw(st.integers(min_value=1, max_value=10))
    num_objects = draw(st.integers(min_value=1, max_value=10))
    object_ids = [f"O{i}" for i in range(num_objects)]

    # Recipes must have produces and/or consumes
    recipes = st.tuples(
        st.sets(st.sampled_from(object_ids)),
        st.sets(st.sampled_from(object_ids)),
    ).filter(lambda x: x[0] or x[1])

    recipe_draws = [
        draw(recipes)
        for i in range(num_processes)
    ]

    processes = [
        # Make sure that the first object is always produced at least once, so
        # it's safe to ask for it
        Process(f"P{i}", consumes=c, produces=(p | {object_ids[0]} if i == 0 else p))
        for i, (c, p) in enumerate(recipe_draws)
    ]

    # Randomly choose if objects have a market if they are both produced and
    # consumed; otherwise it doesn't make sense to.
    obj_produced = set()
    obj_consumed = set()
    for p in processes:
        obj_produced.update(p.produces)
        obj_consumed.update(p.consumes)

    objects = [
        Object(
            object_id,
            MASS,
            has_market=(
                draw(st.booleans()) if (object_id in obj_produced and object_id in obj_consumed)
                else False
            )
        )
        for object_id in object_ids
    ]

    return Model(processes, objects)


@pytest.mark.skip()
@given(model_strategy())
def test_model(m):
    print(m.processes)
    print(m.objects)
    assert m.processes

    # Find an object that we should definitely be able to pull production of
    obj = {x for p in m.processes for x in p.produces}
    # assume(obj)
    obj = obj.pop()

    # Define some allocation coefficients
    allocs = {
        obj.id: {
            p: sy.Dummy() for p in m.producers_of(obj.id)
        }
        for obj in m.objects
    }

    m.add(m.pull_production(obj, sy.Symbol("a"), allocate_backwards=allocs))
    print(m._values)
    print()

    # XXX This assertion is flaky (sometimes passes, sometimes not) and it
    # doesn't test very much anyway -- need to think about what properties to
    # test on the whole model.
    #
    # for j in range(len(m.processes)):
    #     assert m.eval(m.X[j]] == m.eval(m.Y[j]]
