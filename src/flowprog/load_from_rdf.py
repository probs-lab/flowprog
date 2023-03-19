#!/usr/bin/env python3

import logging
log = logging.getLogger(__name__)

from itertools import groupby
from rdflib import Namespace
from probs_runner import PROBS


PROBS_SYS = Namespace("http://ukfires.org/probs/system/")
PROBS_RECIPE = Namespace("https://ukfires.org/probs/ontology/recipe/")
SYS = Namespace("http://c-thru.org/analyses/calculator/system/")
QUANTITYKIND = Namespace("http://qudt.org/vocab/quantitykind/")

NAMESPACES = {
    "sys": SYS,
    "probssys": PROBS_SYS,
    "probs": PROBS,
    "recipe": PROBS_RECIPE,
}


def query_processes(rdfox, model_uri):
    results = rdfox.query_records(
        """
        SELECT ?process ?producesOrConsumes ?recipeObject ?recipeQuantity ?recipeMetric
        WHERE {
            ?model :hasProcess ?process .
            OPTIONAL {
                ?process recipe:hasRecipe ?recipe .
                ?recipe ?producesOrConsumes [ recipe:object ?recipeObject ;
                                              recipe:quantity ?recipeQuantity ;
                                              recipe:metric ?recipeMetric ] .
                FILTER( ?producesOrConsumes = recipe:produces || ?producesOrConsumes = recipe:consumes )
            }
        }
        ORDER BY ?process ?producesOrConsumes ?recipeObject ?recipeQuantity ?recipeMetric
        """,
        initBindings={"model": model_uri},
        initNs=NAMESPACES,
    )

    def _group_to_recipe_list(v, producesOrConsumes):
        items = [
            {
                "object": item["recipeObject"],
                "quantity": item["recipeQuantity"],
                "metric": item["recipeMetric"],
            }
            for item in v
            if item["producesOrConsumes"] == producesOrConsumes
        ]
        if items and any(v is not None for v in items[0].values()):
            return items
        return []

    # Do this in two stages so we can use the generator twice
    grouped = [(k, list(v)) for k, v in groupby(results, lambda d: d["process"])]
    return [
        (
            k,
            _group_to_recipe_list(v, PROBS_RECIPE["produces"]),
            _group_to_recipe_list(v, PROBS_RECIPE["consumes"]),
        )
        for k, v in grouped
    ]


RECIPE_METRIC_UNITS = {
    QUANTITYKIND.Mass: "kg",
    QUANTITYKIND.Area: "m2",
    QUANTITYKIND.Volume: "m3",
    QUANTITYKIND.Dimensionless: "-",
}


def _recipe_items_to_tuples(items):
    for x in items:
        if x["metric"] not in RECIPE_METRIC_UNITS:
            raise ValueError(f"unsupported metric {x['metric']} in recipe")
        assert x["quantity"] is not None, "missing value"

    tuples = [
        (x["object"], RECIPE_METRIC_UNITS[x["metric"]], float(x["quantity"]))
        for x in items
    ]
    return tuples


def _create_recipe_object(item):
    """Define a recipe for process_uri."""
    k, produces, consumes = item
    input_tuples = _recipe_items_to_tuples(consumes)
    output_tuples = _recipe_items_to_tuples(produces)
    return (k, input_tuples, output_tuples)


# def validate_recipes(object_types, defined_recipes):
#     """Validate the recipes -- does everything included in the recipes correspond
#     to a known object?"""
#     obj_dict = {obj.name: obj for obj in object_types}
#     for r in defined_recipes:
#         recipe_name = r.process_uri
#         test_params = {k[0]: test_value(k) for k in getattr(r, "own_params", [])}
#         inputs, outputs = r.recipe(test_params)
#         for k, unit, value in inputs + outputs:
#             if k not in obj_dict:
#                 raise ValueError(f'Unknown object "{k}" in recipe for "{recipe_name}"')
#             if not value >= 0:
#                 raise ValueError(
#                     f'Expect value for "{k}" in recipe for "{recipe_name}" to be positive'
#                 )


def get_recipe_builders(rdfox, model_uri):
    """Query for relevant observations."""
    results = query_processes(rdfox, model_uri)
    recipe_builders = [_create_recipe_object(d) for d in results]
    # validate_recipes(object_types, recipe_builders)
    return recipe_builders


def query_object_types(rdfox, model_uri, object_types):
    query = """
    SELECT ?object ?units ?scale ?isTraded ?hasMarket
    WHERE {
        ?object a :Object .
        OPTIONAL { ?model :hasMarketForObject ?object . BIND(TRUE as ?hasMarket) . }
        OPTIONAL { ?object :objectUnits ?units . }
        OPTIONAL { ?object :objectScale ?scale . }
        OPTIONAL { ?object :objectIsTraded ?isTraded . }
    }
    ORDER BY ?object
    """

    # RDFlib can't handle multiple values
    query += "\nVALUES ( ?model ?object )\n{\n%s\n}\n" % (
        "\n".join(
            "( %s )" % " ".join(
                node.n3()
                for node in [model_uri, object_type]
            )
            for object_type in object_types
        ),
    )

    results = rdfox.query_records(
        query,
        # initBindings={
        #     "model": model_uri,
        #     "object": object_types,
        # },
        initNs=NAMESPACES,
    )

    matched_objects = {obj["object"] for obj in results}
    not_found = object_types - matched_objects
    if not_found:
        raise ValueError("Some objects not found: %s" % not_found)

    return results


def _create_object_type_object(item):
    """Define an object type for query results data."""
    return item


def get_object_types(rdfox, model_uri, object_types):
    """Query for model object types."""
    results = query_object_types(rdfox, model_uri, object_types)
    return [_create_object_type_object(d) for d in results]


def get_objects_from_recipes(recipe_builders):
    """Extract list of all object types that appear in recipes."""
    return {
        obj
        for _, consumes, produces in recipe_builders
        for obj, _, _ in produces + consumes
    }


def strip_uri(uri):
    return str(uri).rpartition("/")[2]


from .imperative_model import Process, Object, Model

def query_model_from_endpoint(rdfox, model_uri, **kwargs):
    """Query to find object types, recipe builders and observations."""

    log.info("Loading recipes...")
    recipe_builders = get_recipe_builders(rdfox, model_uri)

    objects_in_recipes = get_objects_from_recipes(recipe_builders)

    log.info("Loading object types...")
    object_types = get_object_types(rdfox, model_uri, objects_in_recipes)

    # return object_types, recipe_builders

    processes = [
        Process(
            id=strip_uri(k),
            produces=[strip_uri(x) for x, units, value in produces],
            consumes=[strip_uri(x) for x, units, value in consumes],
        )
        for k, consumes, produces in recipe_builders
    ]

    recipes = {
        strip_uri(k): (
            {strip_uri(x): value for x, units, value in consumes},
            {strip_uri(x): value for x, units, value in produces},
        )
        for k, consumes, produces in recipe_builders
    }

    objects = [
        Object(
            id=strip_uri(x["object"]),
            has_market=bool(x["hasMarket"]),
        )
        for x in object_types
    ]

    model = Model(processes, objects, **kwargs)

    recipe_data = {}
    process_idx = {p.id: j for j, p in enumerate(model.processes)}
    object_idx = {o.id: j for j, o in enumerate(model.objects)}
    for p in model.processes:
        j = process_idx[p.id]
        if p.id in recipes:
            consumes, produces = recipes[p.id]
            for obj, value in consumes.items():
                recipe_data[model.U[object_idx[obj], j]] = value
            for obj, value in produces.items():
                recipe_data[model.S[object_idx[obj], j]] = value
        else:
            raise ValueError("No recipe for %s" % p)

    return model, recipe_data
