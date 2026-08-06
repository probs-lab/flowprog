"""Regression tests for the size of expressions produced by the sympy compiler.

Transformations (Limit/Floor) refer to the accumulated model state several
times over -- as 'current', as 'proposed', and as the limit/threshold -- and
their result is then accumulated in turn. Inlining those references made every
step a multiple of the one before, so a handful of limits on the same process
produced expressions of astronomical size, and building them was worse than
linear in that size again: sympy's ExprCondPair runs `piecewise_fold` and
`simplify_logic` over any condition containing a Piecewise, regardless of
`evaluate=False`.

These tests check that compiled expressions stay small, and Piecewise never
appears inside a Piecewise condition.

"""

import pytest
import sympy as sy
from sympy.logic.boolalg import ITE
from rdflib import URIRef

from flowprog.model_builder import ModelBuilder, Process, Object


MASS = URIRef("http://qudt.org/vocab/quantitykind/Mass")

demand = sy.Symbol("demand")
capacity = sy.Symbol("capacity", positive=True)
turndown = sy.Symbol("turndown", positive=True)

RECIPE = {
    "P": {"produces": {"Prod": 0.6}, "consumes": {"Raw": 1.0}},
    "Use": {"produces": {}, "consumes": {"Prod": 1.0}},
}


def make_builder():
    """A one-process chain whose output can be limited over and over."""
    objects = [
        Object("Raw", MASS, has_market=False),
        Object("Prod", MASS, has_market=True),
    ]
    processes = [
        Process("P", produces=["Prod"], consumes=["Raw"], has_stock=False),
        Process("Use", produces=[], consumes=["Prod"], has_stock=False),
    ]
    builder = ModelBuilder(processes, objects)
    builder.add(
        builder.push_process_input("Use", "Prod", demand), label="demand"
    )
    return builder


def add_steps(builder, n, kind="limit"):
    """Add `n` steps that each chase the production deficit of Prod, under a
    transformation reading the state the previous steps accumulated."""
    for i in range(n):
        proposal = builder.pull_process_output(
            "P", "Prod", builder.object_production_deficit("Prod")
        )
        output = builder.expr("ProcessOutput", process_id="P", object_id="Prod")
        if kind == "limit":
            step = builder.limit(proposal, output, capacity)
        elif kind == "floor":
            step = builder.floor(proposal, output, turndown)
        elif kind == "both":
            step = builder.floor(builder.limit(proposal, output, capacity), output, turndown)
        else:
            raise ValueError(kind)
        builder.add(step, label=f"step {i}")
    return builder


def total_ops(model):
    """Total size of everything the compiler produced."""
    return sum(
        sy.count_ops(expr) for expr in model._values.values()
    ) + sum(sy.count_ops(expr) for _, expr, _ in model._intermediates)


@pytest.mark.parametrize("kind", ["limit", "floor", "both"])
def test_repeated_transformations_stay_polynomial(kind):
    """Compiled size must grow polynomially with the number of transformations.

    When the resolved limit expressions were inlined instead of named, size
    grew by a factor of ~7 per step: four steps was already 12k operations and
    took 19 seconds, and six `floor` steps reached 135k operations. It is now
    quadratic -- each step still inlines the accumulated state once, and that
    state carries one term per earlier step.
    """
    small = add_steps(make_builder(), 4, kind).build()
    large = add_steps(make_builder(), 16, kind).build()

    # Quadrupling the steps grows a quadratic ~16x; the bound below leaves
    # room for that while staying far under anything geometric (which would
    # be a ratio of ~7**12 here).
    assert total_ops(large) < 40 * total_ops(small)


def piecewise_conditions(model):
    exprs = list(model._values.values()) + [e for _, e, _ in model._intermediates]
    return [
        cond
        for expr in exprs
        for piecewise in expr.atoms(sy.Piecewise)
        for _, cond in piecewise.args
    ]


@pytest.mark.parametrize("kind", ["limit", "floor", "both"])
def test_conditions_do_not_grow_with_model(kind):
    """Piecewise conditions must stay bounded however many steps precede them.

    This is what keeps sympy off its expensive path. Passing a condition that
    contains a Piecewise to `Piecewise` sends ExprCondPair through
    `piecewise_fold` and `cond.rewrite(ITE)` -- neither of which
    `evaluate=False` avoids -- so the accumulated state gets folded into the
    condition itself. The result is a condition that grows with the model:
    inlining the resolved expressions gave a largest condition of 16, 138 then
    2661 operations over the first four steps. Extracting them as intermediate
    symbols keeps every condition a single relational between two atoms.

    """
    small = piecewise_conditions(add_steps(make_builder(), 4, kind).build())
    large = piecewise_conditions(add_steps(make_builder(), 16, kind).build())
    assert small and large, "expected compiled models to contain Piecewise conditions"

    assert max(sy.count_ops(c) for c in large) == max(sy.count_ops(c) for c in small)


@pytest.mark.parametrize("kind", ["limit", "floor", "both"])
def test_conditions_are_not_rewritten_to_ite(kind):
    """`cond.rewrite(ITE)` is the expensive half of the fold above, and it
    leaves a trace: ITE nodes in the compiled conditions. There should be
    none, because no condition ever contains a Piecewise to fold."""
    model = add_steps(make_builder(), 6, kind).build()

    conditions = piecewise_conditions(model)
    assert conditions
    for cond in conditions:
        assert not cond.atoms(ITE), f"condition rewritten to ITE: {cond}"
        assert not cond.has(sy.Piecewise), f"Piecewise nested in condition: {cond}"


def test_limit_still_applies_after_being_routed_through_intermediates():
    """Naming subexpressions must not change what the limit computes."""
    model = add_steps(make_builder(), 3, "limit").build(RECIPE)
    func = model.lambdify(expressions={"y": model.Y[0]}, modules="math")

    # Demand needs 10/0.6 of P, well over the capacity of 2.
    assert func({"demand": 10.0, "capacity": 2.0})["y"] == pytest.approx(2.0 / 0.6)
    # ... and well under a capacity of 50, so it passes through untouched.
    assert func({"demand": 10.0, "capacity": 50.0})["y"] == pytest.approx(10.0 / 0.6)


def test_floor_still_applies_after_being_routed_through_intermediates():
    model = add_steps(make_builder(), 1, "floor").build(RECIPE)
    func = model.lambdify(expressions={"y": model.Y[0]}, modules="math")

    # Above the turndown threshold the step is kept ...
    assert func({"demand": 10.0, "turndown": 1.0})["y"] == pytest.approx(10.0 / 0.6)
    # ... and below it the process does not operate at all.
    assert func({"demand": 10.0, "turndown": 100.0})["y"] == pytest.approx(0.0)


def test_compiler_intermediates_are_defined_before_use():
    """`_intermediates` is an ordered chain: back-substitution in reverse (as
    `eval_intermediates` does) and forward emission (as `lambdify`'s CSE hook
    does) are both only correct if each definition refers to earlier ones."""
    model = add_steps(make_builder(), 4, "both").build(RECIPE)

    defined: set[sy.Symbol] = set()
    all_syms = {sym for sym, _, _ in model._intermediates}
    for sym, expr, _ in model._intermediates:
        used_intermediates = expr.free_symbols & all_syms
        assert used_intermediates <= defined, (
            f"{sym} refers to {used_intermediates - defined} before they are defined"
        )
        defined.add(sym)


def test_compiler_intermediates_do_not_collide_with_builder_symbols():
    """The compiler mints its own symbols; they must not shadow the ones
    ModelBuilder already put on the steps."""
    builder = add_steps(make_builder(), 4, "limit")
    step_syms = {sym for step in builder._steps for sym, _, _ in step.intermediates}
    model = builder.build(RECIPE)

    seen: set[sy.Symbol] = set()
    for sym, _, _ in model._intermediates:
        assert sym not in seen, f"duplicate intermediate symbol {sym}"
        seen.add(sym)
    minted = seen - step_syms
    assert minted, "expected the compiler to mint intermediates of its own"
    assert not (minted & step_syms)
