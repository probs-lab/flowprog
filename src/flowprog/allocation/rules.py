"""Allocation rules: how a process's burden is split across its flows.

A rule is asked for the weights of the flows of one process at a time, and the
weights are normalised by the caller. See the package docstring for how they
fit together.
"""

from dataclasses import dataclass, field
from typing import Mapping, Protocol

import pandas as pd


class Rule(Protocol):
    """Splits one process's burden across its flows."""

    def weights(self, process, flows: Mapping[str, float]) -> Mapping[str, float]:
        """Raw weights, one for each key of `flows`.

        The weights need not sum to one; they are normalised by the caller. A
        weight of 0 means the flow bears none of the burden.

        :param process: The `Process` whose burden is being split.
        :param flows: ``{object id: quantity}`` -- the flows to split the
            burden across.
        :raises ValueError: If the rule has no value for one of the flows.

        """


def _lookup(values, object_id, process, rule):
    if object_id not in values:
        raise ValueError(
            f"{type(rule).__name__} has no value for {object_id!r}, needed to "
            f"split the burden of process {process.id!r}"
        )
    return values[object_id]


@dataclass
class ByValue:
    """Weights proportional to the flows themselves."""

    def weights(self, process, flows):
        return dict(flows)


@dataclass
class ByProperty:
    """Weights proportional to the flows times a per-object property, such as
    energy content, exergy or price.

    :param properties: ``{object id: property value}``, covering every flow of
        every process this rule is asked about.
    """

    properties: dict

    def weights(self, process, flows):
        return {
            object_id: quantity * _lookup(self.properties, object_id, process, self)
            for object_id, quantity in flows.items()
        }


@dataclass
class Fixed:
    """Explicit weights, independent of the flow quantities.

    :param shares: ``{object id: weight}``, covering every flow of every
        process this rule is asked about.
    """

    shares: dict

    def weights(self, process, flows):
        return {
            object_id: _lookup(self.shares, object_id, process, self)
            for object_id in flows
        }


@dataclass
class Excluding:
    """Objects that bear no burden, with another rule splitting the rest.

    Residual outputs -- air, water, wastes -- usually take no share of the
    burden of the process that emits them::

        Excluding({"Air", "Water", "WasteWater"}, ByValue())

    The excluded objects are removed before `rule` is asked, so `rule` does not
    need data for them.

    :param objects: Object ids that always get weight 0.
    :param rule: Splits the burden across the remaining flows.

    """

    objects: frozenset
    rule: Rule

    def __post_init__(self):
        self.objects = frozenset(self.objects)

    def weights(self, process, flows):
        bearing = {
            object_id: quantity
            for object_id, quantity in flows.items()
            if object_id not in self.objects
        }
        weights = self.rule.weights(process, bearing) if bearing else {}
        return {object_id: weights.get(object_id, 0.0) for object_id in flows}

    def validate(self, structure):
        unknown = self.objects - {o.id for o in structure.objects}
        if unknown:
            raise ValueError(
                f"Excluding references unknown objects {sorted(unknown)}"
            )
        validate_rule(self.rule, structure)


@dataclass
class Rules:
    """A default rule, with exceptions for named processes.

    `Rules` is itself a rule, so it goes anywhere a rule does::

        Rules(
            default=Excluding(RESIDUAL_OUTPUTS, ByValue()),
            by_process={"SteamCracking": ByProperty(energy_content)},
        )

    An entry in `by_process` replaces the default outright for that process:
    each entry states that process's split in full, including which of its
    outputs bear no burden.

    :param default: Rule for processes not named in `by_process`.
    :param by_process: ``{process id: rule}``.
    """

    default: Rule
    by_process: dict = field(default_factory=dict)

    def rule_for(self, process_id) -> Rule:
        """The rule that splits `process_id`'s burden."""
        return self.by_process.get(process_id, self.default)

    def weights(self, process, flows):
        return self.rule_for(process.id).weights(process, flows)

    def validate(self, structure):
        for process_id in self.by_process:
            structure.lookup_process(process_id)  # raises on an unknown id
        validate_rule(self.default, structure)
        for rule in self.by_process.values():
            validate_rule(rule, structure)

    def assignments(self, structure) -> pd.DataFrame:
        """Which rule splits which multi-output process's burden.

        Answers "what will this do?" without running a solve.

        :return: DataFrame with columns ``process``, ``outputs``, ``rule``.
        """
        return pd.DataFrame(
            [
                (p.id, len(p.produces), type(self.rule_for(p.id)).__name__)
                for p in structure.processes
                if len(p.produces) > 1
            ],
            columns=["process", "outputs", "rule"],
        )


def validate_rule(rule, structure):
    """Check a rule's object and process ids against the model, if it can."""
    validate = getattr(rule, "validate", None)
    if validate is not None:
        validate(structure)
