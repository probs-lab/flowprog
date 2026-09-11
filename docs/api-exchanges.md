# Elementary exchanges

Alongside the technosphere `S`/`U` matrices described in {doc}`api-model`,
`flowprog` supports **elementary exchanges**: flows to/from the environment
(CO2, CH4, upstream burdens, ...), represented by the `B` matrix. These are
one-sided flows between a process and the environment. The exchange *types* are
declared via {py:class}`flowprog.ElementaryExchange`; which processes can have
which exchanges is declared per-process (`Process.exchanges`, mirroring
`produces`/`consumes` for `S`/`U`). Values for exchange coefficients are set as
recipe data alongside `S` and `U` (see {py:meth}`SympyModel.set_recipe`,
{py:meth}`ModelBuilder.build`.

## Elementary exchange queries

{py:meth}`ModelBuilder.elementary_balance` returns a structural symbol
`ElementaryBalance` representing the total elementary exchanges at the current
model state, mirroring {py:meth}`ModelBuilder.object_balance`.

`ModelStructure.expr()` understands two relevant roles:
`"ProcessElementaryFlow"` (`exchange_id` + `process_id` -> `B[e,j]*Y[j]`) and
`"ElementaryFlows"` (`exchange_id` alone -> summed over all/specified
processes).

On the evaluable model, {py:meth}`SympyModel.to_elementary_flows` returns a
`(exchange, process, metric, value)` table, similar to
{py:meth}`SympyModel.to_flows`.

## Boundary processes

The `boundary_processes` module provides helpers for building complete process
systems. This is optional: you can build the same thing explicitly by defining
your own `ProductionOfX`, `ImportsOfX`, `ExportsOfX`, etc processes. But since
those process definitions tend to be repetitive, these helpers can be useful. 

It's relevant to elementary exchanges because these processes are often the
place where cradle-to-gate / embodied emissions are linked into the model.

```{eval-rst}
.. automodule:: flowprog.boundary_processes
    :members:
```

## Reporting

```{eval-rst}
.. automodule:: flowprog.reporting
    :members:
```

## Allocation

```{eval-rst}
.. automodule:: flowprog.allocation

.. autoclass:: flowprog.allocation.AllocatedSystem
    :members:

.. autoclass:: flowprog.allocation.Scope
    :members:

.. automodule:: flowprog.allocation.rules
    :members:

.. autoclass:: flowprog.allocation.PassThrough
    :members:
```

## Contribution analysis

An allocated system gives the burden carried by a unit of each object.
Contribution analysis breaks one of those burdens down, by choosing a set of
processes to expand: those are reported with their own direct burdens, and
everything they take in from elsewhere is collapsed into the cradle-to-gate
burden of the input it came in as. Choosing a different set is how the same
product is reported coarsely or in detail, and in more detail along one part of
its supply chain than another.

```{eval-rst}
.. automodule:: flowprog.allocation.contributions
    :members:
```
