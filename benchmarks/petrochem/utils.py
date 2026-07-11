"""Small helpers used by model_polymers.py / model_fertilisers.py.
"""
import sympy as sy


def def_scalar_param(name, **kwargs) -> sy.Symbol:
    """Define a sympy scalar parameter.

    By default assume `nonnegative=True`.
    """
    if "nonnegative" not in kwargs:
        kwargs["nonnegative"] = True
    return sy.Symbol(name, **kwargs)


def def_vector_param(name, **kwargs) -> sy.IndexedBase:
    """Define a sympy vector parameter.

    By default assume `nonnegative=True`.
    """
    if "nonnegative" not in kwargs:
        kwargs["nonnegative"] = True
    return sy.IndexedBase(name, **kwargs)


def pull_production_with_capacity_limit(
    model, object_id, limit_object, limit_processes, capacity, **kwargs
):
    """Helper to set up 'pull production' with a capacity limit."""

    # Initial proposal for required production, before considering the capacity limit
    proposal = model.pull_production(
        object_id, model.object_production_deficit(object_id), **kwargs
    )

    # Limit by capacity
    limited = model.limit(
        proposal,
        model.expr(
            "SoldProduction", object_id=limit_object, limit_to_processes=limit_processes
        ),
        capacity,
    )

    return limited
