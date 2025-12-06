"""Shared hypothesis strategies and helpers for property-based testing of flowprog models."""

from hypothesis import strategies as st
from hypothesis import assume
from rdflib import URIRef
from flowprog.imperative_model import Model, Process, Object


# Shorthand for creating objects
MASS = URIRef("http://qudt.org/vocab/quantitykind/Mass")


def MObject(id, *args, **kwargs):
    """Shorthand for creating an Object with MASS metric."""
    return Object(id, MASS, *args, **kwargs)


def has_cycle_through_market(processes, objects):
    """Check if there's a cycle that passes through an object with has_market=True.

    A cycle only matters if it goes through at least one object that has a market.
    This is because objects without markets don't create problematic feedback loops.
    """
    # Build object ID to has_market mapping
    market_objects = {obj.id for obj in objects if obj.has_market}

    # If no market objects, cycles don't matter
    if not market_objects:
        return False

    # Build directed graph: object -> objects reachable through processes
    graph = {}  # object_id -> set of object_ids
    all_obj_ids = {obj.id for obj in objects}

    for obj_id in all_obj_ids:
        graph[obj_id] = set()

    for proc in processes:
        # For each process, add edges from consumed objects to produced objects
        for consumed in proc.consumes:
            for produced in proc.produces:
                graph[consumed].add(produced)

    # DFS-based cycle detection using three-color algorithm
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {obj_id: WHITE for obj_id in all_obj_ids}

    def dfs(node, path):
        """Perform DFS to detect cycles. Returns True if a cycle through a market object is found."""
        if color[node] == BLACK:
            # Already fully explored
            return False
        if color[node] == GRAY:
            # Found a cycle - check if it goes through a market object
            cycle_start = path.index(node)
            cycle = path[cycle_start:]
            return any(obj_id in market_objects for obj_id in cycle)

        color[node] = GRAY
        path.append(node)

        for neighbor in graph[node]:
            if dfs(neighbor, path):
                return True

        path.pop()
        color[node] = BLACK
        return False

    # Check for cycles starting from each unvisited node
    for obj_id in all_obj_ids:
        if color[obj_id] == WHITE:
            if dfs(obj_id, []):
                return True

    return False


@st.composite
def model_strategy(
    draw,
    min_processes=1,
    max_processes=10,
    min_objects=1,
    max_objects=10,
    randomize_has_stock=False,
):
    """Generate a random model structure (processes and objects).

    Args:
        draw: Hypothesis draw function
        min_processes: Minimum number of processes (default 1)
        max_processes: Maximum number of processes (default 10)
        min_objects: Minimum number of objects (default 1)
        max_objects: Maximum number of objects (default 10)
        randomize_has_stock: If True, randomize Process.has_stock (default False)

    Returns:
        Model instance with generated structure
    """
    num_processes = draw(st.integers(min_value=min_processes, max_value=max_processes))
    num_objects = draw(st.integers(min_value=min_objects, max_value=max_objects))
    object_ids = [f"O{i}" for i in range(num_objects)]

    # Recipes must have produces and/or consumes
    # Also ensure a process doesn't both produce and consume the same object
    recipes = st.tuples(
        st.sets(st.sampled_from(object_ids), max_size=4),
        st.sets(st.sampled_from(object_ids), max_size=4),
    ).filter(lambda x: (x[0] or x[1]) and x[0].isdisjoint(x[1]))

    recipe_draws = [draw(recipes) for i in range(num_processes)]

    processes = [
        # Make sure that the first object is always produced at least once, so
        # it's safe to ask for it
        Process(
            f"P{i}",
            consumes=list(c),
            produces=list(p | ({object_ids[0]} if i == 0 else set())),
            has_stock=draw(st.booleans()) if randomize_has_stock else False,
        )
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
                draw(st.booleans())
                if (object_id in obj_produced and object_id in obj_consumed)
                else False
            ),
        )
        for object_id in object_ids
    ]

    # Reject models with cycles through market objects
    assume(not has_cycle_through_market(processes, objects))

    return Model(processes, objects)
