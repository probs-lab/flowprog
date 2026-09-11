"""Allocation: average burdens carried by each object and process, and where
those burdens came from.

An :class:`AllocatedSystem` is a linear representation of a model at one
operating point, with allocation rules applied. Building one evaluates the
model, splits each process's burden across its flows, and solves for the burden
carried by everything in it::

    allocated = AllocatedSystem(model, params, rule=ByValue())
    allocated.object_intensities     # burden per unit of each object
    allocated.process_intensities    # burden per unit of each process's activity
    allocated.object_burdens         # and the totals at this operating point

To partition one of those burdens, :meth:`AllocatedSystem.contributions` breaks
it into the direct emissions of a set of expanded processes plus the collapsed
upstream burden of everything else::

    result = allocated.contributions(
        object="Polymer",
        expand=ProcessSet.producers_of(model.structure, "Polymer"),
    )
    result.direct        # emitted by the expanded processes
    result.upstream      # collapsed burden of what they took in
    result.by("object")  # both together, by input object

See :mod:`flowprog.allocation.contributions` for what expanding means and how
to choose the processes.

Allocation rules
----------------

Each process's burden is split across its outputs using a *rule*, which is
asked for the weights of one process's flows at a time and normalised by the
caller::

    ByValue()                   proportional to the output quantities
    ByProperty(properties)      ... times a per-object property
    Fixed(shares)               explicit weights
    Excluding(objects, rule)    those objects bear no burden
    Rules(default, by_process)  a default, with exceptions per process

`Rules` is itself a rule, so the pieces nest::

    Rules(
        default=Excluding({"Air", "Water", "WasteWater"}, ByValue()),
        by_process={
            "CombinedHeatAndPower": Fixed({"Electricity": 0.6, "Heat": 0.4}),
        },
    )

Stocks
------

A process with `has_stock` can take in more than it puts out, or less (``X !=
Y``). By default, output flows carry burdens based on the input flows scaled to
the output activity, so material flowing out has the same burden intensity as
the material flowing in. The difference in absolute burden is reported as
`AllocatedSystem.stock_burden`.

Pass-through
------------

:class:`PassThrough` is a separate, special-case calculation: a designated set
of processes have their burdens pushed to their direct consumers in proportion
to consumption, and every other process retains its own. Because burden is only
moved, any grouping over the reattributed table still partitions the system
total. It is implemented symbolically, so the resulting expressions can be
compiled to standalone code.

"""

from .contributions import ContributionResult, ProcessSet
from .passthrough import PassThrough
from .rules import ByProperty, ByValue, Excluding, Fixed, Rule, Rules
from .system import AllocatedSystem, Scope

__all__ = [
    "AllocatedSystem",
    "ByProperty",
    "ByValue",
    "ContributionResult",
    "Excluding",
    "Fixed",
    "PassThrough",
    "ProcessSet",
    "Rule",
    "Rules",
    "Scope",
]
