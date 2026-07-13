"""Boundary processes: declarative expansion of import/export/source/sink specs
into ordinary explicit processes carrying elementary-exchange (B) burdens.

"Boundary" means the boundary of the *modelled* system, not a geographic
boundary. These are the processes at which propagation stops and markets
close -- the standard way upstream/downstream emission factors enter a model.

There are two axes describing the boundary process being added: a *structural*
axis (does the process supply or remove the object?) and a *semantic* axis
(trade across the boundary vs. a source/sink term within it). The semantic axis
does not change anything structural, and only sets default naming.

"""

import logging
from dataclasses import dataclass, field
from typing import Iterable, Literal, Union

import sympy as sy

from .model_structure import ModelStructure, Process

_log = logging.getLogger(__name__)

Kind = Literal["import", "export", "source", "sink"]

# kind -> (direction, default process id prefix)
_KIND_TABLE: dict[Kind, tuple[str, str]] = {
    "import": ("supply", "ImportsOf"),
    "export": ("removal", "ExportsOf"),
    "source": ("supply", "SourceOf"),
    "sink": ("removal", "SinkOf"),
}

# A recipe fragment in the same ID-based format accepted by
# ``SympyModel.set_recipe`` / ``ModelBuilder.build(recipe_data=...)``.
RecipeFragment = dict[str, dict]


@dataclass(frozen=True)
class BoundaryProcess:
    """Specification for a process that crosses the boundary of the modelled system.

    :param object_id: The technosphere object supplied or removed.
    :param kind: One of "import", "export", "source", "sink". "import"/"export"
        describe trade across the system boundary; "source"/"sink" describe an
        unmodelled term within it. Both "import" and "source" are
        supply-side (the generated process produces `object_id`); both
        "export" and "sink" are removal-side (the generated process consumes
        `object_id`).
    :param exchanges: Elementary exchange coefficients per unit of object
        supplied/removed, e.g. ``{"GHG_upstream_CO2e": 423}``.
    :param process_id: Optional explicit id for the generated process.
        Defaults to a kind-specific prefix + object_id, e.g. "ImportsOfNaphtha".
    """

    object_id: str
    kind: Kind
    exchanges: dict[str, Union[sy.Expr, float]] = field(default_factory=dict)
    process_id: Union[str, None] = None

    def __post_init__(self):
        if self.kind not in _KIND_TABLE:
            raise ValueError(
                f"Unknown boundary process kind {self.kind!r}; "
                f"must be one of {sorted(_KIND_TABLE)}"
            )

    @property
    def direction(self) -> str:
        """Either "supply" (process produces `object_id`) or "removal" (consumes it)."""
        return _KIND_TABLE[self.kind][0]

    def resolved_process_id(self) -> str:
        """The id of the generated process: `process_id` if given, else a default."""
        if self.process_id is not None:
            return self.process_id
        prefix = _KIND_TABLE[self.kind][1]
        return f"{prefix}{self.object_id}"


def Import(object_id: str, **kwargs) -> BoundaryProcess:
    """Shorthand for ``BoundaryProcess(object_id, "import", **kwargs)``."""
    return BoundaryProcess(object_id, "import", **kwargs)


def Export(object_id: str, **kwargs) -> BoundaryProcess:
    """Shorthand for ``BoundaryProcess(object_id, "export", **kwargs)``."""
    return BoundaryProcess(object_id, "export", **kwargs)


def Source(object_id: str, **kwargs) -> BoundaryProcess:
    """Shorthand for ``BoundaryProcess(object_id, "source", **kwargs)``."""
    return BoundaryProcess(object_id, "source", **kwargs)


def Sink(object_id: str, **kwargs) -> BoundaryProcess:
    """Shorthand for ``BoundaryProcess(object_id, "sink", **kwargs)``."""
    return BoundaryProcess(object_id, "sink", **kwargs)


def add_boundary_processes(
    structure: ModelStructure, specs: Iterable[BoundaryProcess]
) -> tuple[ModelStructure, RecipeFragment]:
    """Expand boundary process specs into an enlarged structure + recipe fragment.

    Pure structure-to-structure pre-processing: run before the builder. The
    returned structure has one new :class:`Process` per spec (always
    ``has_stock=False``); the returned recipe fragment carries the S=1/U=1
    technosphere coefficient and the declared B entries for each generated
    process, in the ID-based recipe format. Merge it into your recipe data
    before calling ``ModelBuilder.build()`` (or call
    ``model.set_recipe(fragment)`` after building).

    This does not introduce any model logic (when/how these processes actually
    supply or absorb flow) -- you need to ensure the boundary processes are
    actually used in a model builder step, just like any other process.

    :param structure: Existing model structure to extend
    :param specs: Boundary process specifications
    :return: (enlarged ModelStructure, recipe fragment)
    :raises ValueError: If an object/exchange id is unknown, or a generated
        process id collides with an existing or another generated process

    """
    specs = list(specs)

    existing_process_ids = {p.id for p in structure.processes}
    new_processes = list(structure.processes)
    recipe_fragment: RecipeFragment = {}
    spec_by_process_id: dict[str, BoundaryProcess] = {}

    for spec in specs:
        i = structure.lookup_object(spec.object_id)
        obj = structure.objects[i]
        if not obj.has_market:
            _log.warning(
                "Boundary process %r targets object %r which does not have a "
                "market (has_market=False) -- this is almost certainly a mistake.",
                spec.kind,
                spec.object_id,
            )

        for exchange_id in spec.exchanges:
            structure.lookup_exchange(exchange_id)

        process_id = spec.resolved_process_id()
        if process_id in existing_process_ids or process_id in spec_by_process_id:
            raise ValueError(
                f"Boundary process id {process_id!r} collides with an existing "
                "or another generated process"
            )

        if spec.direction == "supply":
            new_processes.append(
                Process(
                    id=process_id,
                    produces=[spec.object_id],
                    consumes=[],
                    has_stock=False,
                )
            )
            recipe_fragment[process_id] = {
                "produces": {spec.object_id: 1},
                "consumes": {},
                "exchanges": dict(spec.exchanges),
            }
        else:
            new_processes.append(
                Process(
                    id=process_id,
                    produces=[],
                    consumes=[spec.object_id],
                    has_stock=False,
                )
            )
            recipe_fragment[process_id] = {
                "produces": {},
                "consumes": {spec.object_id: 1},
                "exchanges": dict(spec.exchanges),
            }

        spec_by_process_id[process_id] = spec

    new_structure = ModelStructure(
        new_processes, structure.objects, structure.elementary_exchanges
    )
    # Expansion metadata, retrievable for reporting grouping defaults.
    new_structure.boundary_process_specs = spec_by_process_id

    return new_structure, recipe_fragment
