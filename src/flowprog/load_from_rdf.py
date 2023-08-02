#!/usr/bin/env python3

import logging
log = logging.getLogger(__name__)

from itertools import groupby
from rdflib import Namespace


PROBS_SYS = Namespace("http://ukfires.org/probs/system/")
PROBS_RECIPE = Namespace("https://ukfires.org/probs/ontology/recipe/")
PROBS = Namespace("https://ukfires.org/probs/ontology/")
SYS = Namespace("http://c-thru.org/analyses/calculator/system/")
QUANTITYKIND = Namespace("http://qudt.org/vocab/quantitykind/")

NAMESPACES = {
    "sys": SYS,
    "probssys": PROBS_SYS,
    "probs": PROBS,
    "recipe": PROBS_RECIPE,
}


def query_processes(rdfox, model_uri):
    results = rdfox.query(
        """
        SELECT DISTINCT ?process ?producesOrConsumes ?recipeObject ?recipeQuantity ?recipeMetric
        WHERE {
            ?model probs:hasProcess ?process .
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


def _recipe_items_to_tuples(items):
    for x in items:
        assert x["quantity"] is not None, "missing value"

    tuples = [
        (x["object"], x["metric"], float(x["quantity"]))
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


def query_object_types(rdfox, model_uri):
    query = """
    SELECT ?object ?metric ?isTraded ?hasMarket
    WHERE {
        # First identify relevant object types based on inputs/outputs of processes
        {
            SELECT DISTINCT ?object
            WHERE {
                ?model probs:hasProcess ?process .
                ?process recipe:hasRecipe ?recipe .
                ?recipe ?producesOrConsumes [ recipe:object ?object ] .
                FILTER( ?producesOrConsumes = recipe:produces || ?producesOrConsumes = recipe:consumes ) .
            }
        }

        # Then lookup information about each object
        OPTIONAL { ?model probs:hasMarketForObject ?object . BIND(1 as ?hasMarket) . }
        OPTIONAL { ?object probs:objectMetric ?metric . }
        OPTIONAL { ?object probs:objectIsTraded ?isTraded . }
    }
    ORDER BY ?object
    """

    results = rdfox.query(
        query,
        initBindings={
            "model": model_uri,
        },
        initNs=NAMESPACES,
    )

    # XXX not sure what to check to verify all have been found?
    # matched_objects = {obj["object"] for obj in results}
    # not_found = object_types - matched_objects
    # if not_found:
    #     raise ValueError("Some objects not found: %s" % not_found)

    return results


def _create_object_type_object(item):
    """Define an object type for query results data."""
    return item


def get_object_types(rdfox, model_uri):
    """Query for model object types."""
    results = query_object_types(rdfox, model_uri)
    return [_create_object_type_object(d) for d in results]


def strip_uri(uri):
    return str(uri).rpartition("/")[2]


from .imperative_model import Process, Object, Model

def query_model_from_endpoint(rdfox, model_uri, **kwargs) -> tuple[Model, dict]:
    """Query to find object types, recipe builders and observations."""

    log.info("Loading recipes...")
    recipe_builders = get_recipe_builders(rdfox, model_uri)

    log.info("Loading object types...")
    object_types = get_object_types(rdfox, model_uri)

    # return object_types, recipe_builders

    processes = [
        Process(
            id=strip_uri(k),
            produces=[strip_uri(x) for x, metric, value in produces],
            consumes=[strip_uri(x) for x, metric, value in consumes],
        )
        for k, consumes, produces in recipe_builders
    ]

    # XXX can neaten this up with better treatment of prefixes/uris
    recipes = {
        strip_uri(k): (
            {strip_uri(x): (metric, value) for x, metric, value in consumes},
            {strip_uri(x): (metric, value) for x, metric, value in produces},
        )
        for k, consumes, produces in recipe_builders
    }

    objects = [
        Object(
            id=strip_uri(x["object"]),
            metric=x["metric"] or QUANTITYKIND.Mass,
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
            for obj, (metric, value) in consumes.items():
                i = object_idx[obj]
                expected_metric = objects[i].metric
                if metric != expected_metric:
                    raise ValueError("Expected metric %r for %r but got %r" % (str(expected_metric), str(obj), str(metric)))
                recipe_data[model.U[i, j]] = value
            for obj, (metric, value) in produces.items():
                i = object_idx[obj]
                expected_metric = objects[i].metric
                if metric != expected_metric:
                    raise ValueError("Expected metric %r for %r but got %r" % (str(expected_metric), str(obj), str(metric)))
                recipe_data[model.S[i, j]] = value
        else:
            raise ValueError("No recipe for %s" % p)

    return model, recipe_data
