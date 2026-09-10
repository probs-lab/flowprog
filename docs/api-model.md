# Model building API

The structure of a model is defined by the `Process`es and `Object`s it contains, collectively the {py:class}`ModelStructure`.  Then the equations are built up step-by-step by a {py:class}`ModelBuilder`.

When complete, the model steps, optionally bundled with some fixed recipe data, are compiled to produce a finished model than can be evaluated with different parameter values. Currently there is a {py:class}`SympyModel` and a {py:class}`NumpyroModel` backend.

## Model structure

```{eval-rst}
.. autoclass:: flowprog.Process
    :members:

.. autoclass:: flowprog.Object
    :members:

.. autoclass:: flowprog.ElementaryExchange
    :members:

.. autoclass:: flowprog.ModelStructure
    :members:
```

Elementary exchanges (`CO2`, `CH4`, upstream `GHG_upstream_CO2e`-style aggregates, ...)
represent flows to/from the environment, alongside the technosphere objects above --
see {doc}`api-exchanges` for exchanges, boundary processes, reporting, and allocation.


## Model building

The idea is to

1. Start with Sympy symbols representing the unknown parameters of the model (e.g. demand of a set of products).
2. Use methods such as {py:meth}`~flowprog.ModelBuilder.pull_production` to find a set of updated equations for various system variables.
3. Optionally, modify these updated equations using methods such as {py:meth}`~flowprog.ModelBuilder.limit`.
4. Add the updated equations into the model using {py:meth}`~flowprog.ModelBuilder.add`, meaning they will now be taken into account for later calculations.

When a complete model has been built, a snapshot is taken and returned as the {py:class}`~flowprog.SympyModel` returned by {py:meth}`ModelBuilder.build <flowprog.ModelBuilder.build>`.  From this, expressions for the flows can be exported using {py:meth}`SympyModel.to_flows <flowprog.SympyModel.to_flows>`. These can either be saved for later use still containing placeholder symbols (e.g. to be filled in later depending on user-specified scenarios or parameters) or can be substituted for specific values directly.  For repeated evaluation the model can be compiled using {py:meth}`~flowprog.SympyModel.lambdify`.

```{eval-rst}
.. autoclass:: flowprog.ModelBuilder
    :members:
```

(api-checking-balance)=
## Checking that markets balance


```{eval-rst}
.. automodule:: flowprog.balance

.. py:class:: flowprog.balance.Verdict

   .. autoattribute:: flowprog.balance.Verdict.BALANCED
   .. autoattribute:: flowprog.balance.Verdict.CONDITIONAL
   .. autoattribute:: flowprog.balance.Verdict.OPEN

.. autoclass:: flowprog.BalanceTrace
    :members:

.. autoclass:: flowprog.balance.BalanceEvent
    :members:

.. py:class:: flowprog.balance.Effect

   The effect of one step on one object's balance.

   .. autoattribute:: flowprog.balance.Effect.OPENS
   .. autoattribute:: flowprog.balance.Effect.CLOSES
   .. autoattribute:: flowprog.balance.Effect.CLOSES_UNLESS_CONSTRAINED

.. py:class:: flowprog.balance.Sign

   The sign of an expression, as far as it can be determined.

   .. autoattribute:: flowprog.balance.Sign.ZERO
   .. autoattribute:: flowprog.balance.Sign.NON_NEGATIVE
   .. autoattribute:: flowprog.balance.Sign.NON_POSITIVE
   .. autoattribute:: flowprog.balance.Sign.UNKNOWN
   
```

## SympyModel instances

```{eval-rst}
.. autoclass:: flowprog.SympyModel
    :members:
```

## NumpyroModel instances

```{eval-rst}
.. autoclass:: flowprog.NumpyroModel
    :members:
```
