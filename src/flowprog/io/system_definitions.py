"""Load a model structure directly from *system definitions*.

System definitions are documents (MyST markdown or reStructuredText) that describe a
system using ``system:process`` / ``system:object`` directives, in the notation
defined by `sphinx_probs_rdf <https://github.com/ricklupton/sphinx_probs_rdf>`_::

    ```{system:process} Disassembly
    ---
    consumes: |
      EoLTechnology                            = 1 kg
    produces: |
      PCBs                                     = 0.2 kg
      OtherParts                               = 0.8 kg
    ---
    ```

:mod:`flowprog.io.rdf` reads the same information after a Sphinx build has exported
it to RDF. This module parses the source documents directly instead, so a model can
be re-run as soon as its definitions are edited.

Which processes make up the model
---------------------------------

The RDF path takes this from a separate model definition (``probs:hasProcess`` and
``probs:hasMarketForObject``). System definitions carry no such statement, so by
default it is inferred:

- the model's **processes** are those that define a recipe -- parent processes that
  only group their children, and declare inputs and outputs without amounts, are
  left out (see `select_processes`);
- an object gets a **market** if the model both produces and consumes it, and so has
  to balance it. An object only produced, or only consumed, is outside the model
  boundary -- an elementary flow, in LCA terms (see `select_markets`).

Pass `processes` or `markets` to `load_structure` to override either.

Recipes written as shares
-------------------------

A recipe side written in bare ``%`` is loaded when every object in that process's
recipe is measured in the same metric: the shares are then a dimensionless recipe in
that metric, and are simply divided by 100. ``Plastic = 40 %`` in an all-mass process
means the same as ``Plastic = 0.4 kg`` per unit of process activity.

Shares are rejected when the process mixes metrics -- relating a side in kg to one in
m3 needs a coefficient in m3/kg, which a share cannot express -- and ``%basis`` shares
are rejected outright, since they name a basis to convert into.

.. warning::

    Parsing a system definition executes any ``:defs:`` blocks and recipe amount
    expressions it contains. Only parse definitions you trust.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Container, Iterable, Mapping, Optional, Sequence, Union

from rdflib import Namespace, URIRef

from ..model_structure import ModelStructure, Object, Process

try:
    from sphinx_probs_rdf.loader import parse_system_definitions
    from sphinx_probs_rdf.model import ParsedSystem, ProcessDef, RecipeItem
    from sphinx_probs_rdf.units import DEFAULT_UNITS, Unit, UnitTable
except ImportError as err:  # pragma: no cover - depends on what is installed
    raise ImportError(
        "flowprog.io.system_definitions requires sphinx_probs_rdf with its `loader` "
        "extra; install `flowprog[definitions]` to get it."
    ) from err


log = logging.getLogger(__name__)


QUANTITYKIND = Namespace("http://qudt.org/vocab/quantitykind/")

#: Used for an object whose basis could not be determined, matching
#: :mod:`flowprog.io.rdf`.
DEFAULT_METRIC = QUANTITYKIND.Mass

PathLike = Union[str, Path]


def unit_table(
    units: Mapping[str, Union[str, tuple[float, str]]],
    *,
    basis_prefix: str = str(QUANTITYKIND),
    prefixes: Optional[Mapping[str, str]] = None,
) -> UnitTable:
    """Build a `UnitTable` covering `units` as well as the default kg/m2/m3/-.

    `units` is written the same way as Sphinx's ``probs_rdf_units`` setting, mapping
    a unit to its basis, optionally with a scale::

        unit_table({"kWh": (1, "Energy"), "pkm": "<http://example.org/PassengerKM>"})

    A basis written as ``<uri>`` is used as-is, ``prefix:name`` is resolved against
    `prefixes`, and a bare name is resolved against `basis_prefix` -- QUDT quantity
    kinds unless the definitions use their own basis vocabulary.
    """
    table = dict(DEFAULT_UNITS.units)
    for unit, value in units.items():
        scale, basis = (1.0, value) if isinstance(value, str) else value
        table[unit] = Unit(float(scale), _resolve_basis(basis, basis_prefix, prefixes))
    return UnitTable(units=table)


def _resolve_basis(
    basis: str, basis_prefix: str, prefixes: Optional[Mapping[str, str]]
) -> str:
    if basis.startswith("<") and basis.endswith(">"):
        return basis[1:-1]
    prefix, _, name = basis.rpartition(":")
    if not name:
        raise ValueError(f"missing name in basis {basis!r}")
    if not prefix:
        return basis_prefix + name
    if not prefixes or prefix not in prefixes:
        raise ValueError(f"unknown prefix {prefix!r} in basis {basis!r}")
    return prefixes[prefix] + name


def select_processes(system: ParsedSystem) -> list[str]:
    """Return the names of `system`'s processes that define a recipe.

    Processes that only declare which objects they involve, without amounts, are
    excluded: they group their children rather than describing a flow themselves.
    """
    return sorted(system.processes_with_recipes())


def select_markets(system: ParsedSystem, processes: Iterable[str]) -> list[str]:
    """Return the objects that `processes` both produce and consume.

    These are the objects whose supply and demand the model has to balance; the rest
    cross the model boundary and are left unbalanced.
    """
    produced: set[str] = set()
    consumed: set[str] = set()
    for name in processes:
        process = system.processes[name]
        produced.update(item.object_name for item in process.produces)
        consumed.update(item.object_name for item in process.consumes)
    return sorted(produced & consumed)


def structure_from_parsed_system(
    system: ParsedSystem,
    *,
    processes: Optional[Sequence[str]] = None,
    markets: Optional[Sequence[str]] = None,
) -> tuple[ModelStructure, dict]:
    """Build a `ModelStructure` from an already-parsed `system`.

    :param system: A `sphinx_probs_rdf.model.ParsedSystem`.
    :param processes: Names of the processes making up the model. Defaults to
        `select_processes(system)`.
    :param markets: Names of the objects to balance. Defaults to
        `select_markets(system, processes)`.
    :return: `(structure, recipe_data)`, where `recipe_data` maps the structure's
        `U`/`S` symbols to recipe coefficients.

    Processes, objects, and each process's inputs and outputs are ordered by name, so
    the same definitions always give the same structure.
    """
    process_names = (
        select_processes(system)
        if processes is None
        else _check_known(processes, system.processes, "process")
    )

    object_names = sorted(
        {
            item.object_name
            for name in process_names
            for item in _recipe_items(system.processes[name])
        }
    )
    _check_known(object_names, system.objects, "object")

    market_names = set(
        select_markets(system, process_names)
        if markets is None
        else _check_known(markets, set(object_names), "market object")
    )

    metrics = _metrics(system, process_names, object_names)
    objects = [
        Object(id=name, metric=metrics[name], has_market=name in market_names)
        for name in object_names
    ]
    structure = ModelStructure(
        [
            Process(
                id=name,
                consumes=sorted(i.object_name for i in system.processes[name].consumes),
                produces=sorted(i.object_name for i in system.processes[name].produces),
            )
            for name in process_names
        ],
        objects,
    )

    recipe_data = {}
    for name in process_names:
        process = system.processes[name]
        j = structure.lookup_process(name)
        common_metric = _common_metric(structure, objects, process)
        for symbol, items in (
            (structure.U, process.consumes),
            (structure.S, process.produces),
        ):
            for item in items:
                i = structure.lookup_object(item.object_name)
                key = symbol[i, j]
                if key in recipe_data:
                    raise ValueError(
                        f"{process.source}: process {name!r} lists "
                        f"{item.object_name!r} more than once on the same side"
                    )
                recipe_data[key] = _quantity(
                    system, process, item, objects[i].metric, common_metric
                )

    return structure, recipe_data


def load_structure(
    paths: Union[PathLike, Iterable[PathLike]],
    *,
    processes: Optional[Sequence[str]] = None,
    markets: Optional[Sequence[str]] = None,
    units: Optional[UnitTable] = None,
    validate: bool = True,
) -> tuple[ModelStructure, dict]:
    """Load a `ModelStructure` from the system definitions in `paths`.

    :param paths: System definition documents, parsed in the order given. A single
        path may be passed on its own.
    :param processes: Names of the processes making up the model. Defaults to those
        that define a recipe -- see `select_processes`.
    :param markets: Names of the objects to balance. Defaults to those the model both
        produces and consumes -- see `select_markets`.
    :param units: Units the definitions may use, as built by `unit_table`. Defaults
        to kg/m2/m3/- measured in QUDT quantity kinds.
    :param validate: Whether to log warnings about problems `sphinx_probs_rdf` finds
        in the definitions themselves, such as references to undeclared objects.
    :return: `(structure, recipe_data)`, ready to pass to
        :meth:`~flowprog.model_builder.ModelBuilder.from_structure` and
        :meth:`~flowprog.model_builder.ModelBuilder.build` respectively.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    system = parse_system_definitions(
        paths, units=units if units is not None else DEFAULT_UNITS, validate=validate
    )
    return structure_from_parsed_system(system, processes=processes, markets=markets)


def _recipe_items(process: ProcessDef) -> list[RecipeItem]:
    return process.consumes + process.produces


def _check_known(names: Iterable[str], known: Container[str], what: str) -> list[str]:
    names = sorted(set(names))
    unknown = [name for name in names if name not in known]
    if unknown:
        raise ValueError(
            "unknown %s%s: %s" % (what, "s" if len(unknown) > 1 else "", unknown)
        )
    return names


def _metrics(
    system: ParsedSystem, process_names: Iterable[str], object_names: Iterable[str]
) -> dict[str, URIRef]:
    """Work out the metric to measure each of `object_names` in.

    `sphinx_probs_rdf` infers an object's basis from the units its recipe items are
    written in, which leaves an object appearing only in shares without one. A bare
    share means "a fraction of this side's common basis", so such an object is
    measured in whatever metric the processes using it are written in.
    """
    metrics = {
        name: URIRef(system.objects[name].basis)
        for name in object_names
        if system.objects[name].basis is not None
    }

    unknown = [name for name in object_names if name not in metrics]
    if not unknown:
        return metrics

    from_processes: dict[str, set[URIRef]] = {name: set() for name in unknown}
    for process_name in process_names:
        items = _recipe_items(system.processes[process_name])
        involved = {item.object_name for item in items}
        known = {metrics[name] for name in involved if name in metrics}
        for name in involved & from_processes.keys():
            from_processes[name] |= known

    for name in unknown:
        candidates = from_processes[name]
        if len(candidates) == 1:
            metrics[name] = candidates.pop()
        else:
            log.warning(
                "No basis for object %r%s, assuming %s",
                name,
                (
                    " and the processes using it disagree on one"
                    if candidates
                    else ""
                ),
                DEFAULT_METRIC,
            )
            metrics[name] = DEFAULT_METRIC
    return metrics


def _common_metric(
    structure: ModelStructure, objects: list[Object], process: ProcessDef
) -> Optional[URIRef]:
    """The metric shared by every object in `process`'s recipe, if there is one."""
    metrics = {
        objects[structure.lookup_object(item.object_name)].metric
        for item in _recipe_items(process)
    }
    return metrics.pop() if len(metrics) == 1 else None


def _quantity(
    system: ParsedSystem,
    process: ProcessDef,
    item: RecipeItem,
    metric: URIRef,
    common_metric: Optional[URIRef],
) -> float:
    where = f"{process.source}: process {process.name!r} item {item.object_name!r}"
    if item.amount is None:
        raise ValueError(
            f"{where} has no amount. Processes without a full recipe cannot be part "
            "of the model -- did you mean to select only their children?"
        )
    quantity = item.quantity(system.units)
    assert quantity is not None  # amount implies unit, so this is always resolved

    if item.is_share:
        if item.share_layer is not None:
            raise NotImplementedError(
                f"{where} is a share in a named basis ({item.unit}), which implies a "
                f"conversion factor into {item.share_layer!r} that flowprog cannot "
                "apply. Write the recipe as amounts instead."
            )
        if common_metric is None:
            raise NotImplementedError(
                f"{where} is a share ({item.unit}), but process {process.name!r} does "
                "not measure everything in its recipe in one metric, so a share does "
                "not determine the amount. Write the recipe as amounts instead."
            )
        # A share of a recipe whose objects are all in one metric is just a
        # dimensionless coefficient in that metric, per unit of process activity.
        return quantity / 100.0

    basis = item.basis(system.units)
    if basis is not None and URIRef(basis) != metric:
        raise ValueError(
            f"{where} is measured in {basis!r}, but object {item.object_name!r} has "
            f"metric {str(metric)!r}"
        )
    return quantity
