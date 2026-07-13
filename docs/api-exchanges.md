# Elementary exchanges

Alongside the technosphere `S`/`U` matrices described in {doc}`api-model`,
`flowprog` supports **elementary exchanges**: flows to/from the environment
(CO2, CH4, upstream burdens, ...), represented by the `B` matrix. These are
one-sided flows between a process and the environment, declared via
{py:class}`flowprog.ElementaryExchange` and set as recipe data alongside `S` and
`U` (see {py:meth}`SympyModel.set_recipe`, {py:meth}`ModelBuilder.build`).

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
    :members:
```
