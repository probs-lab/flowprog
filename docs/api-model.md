# Model building API

The structure of a model is defined by the `Process`es and `Object`s it contains, collectively the {py:class}`ModelStructure`.  Then the equations are built up by a {py:class}`ModelBuilder`.  The complete model, optionally bundled with some fixed recipe data, is the {py:class}`Model`.

## Model structure

```{eval-rst}
.. autoclass:: flowprog.Process
    :members:

.. autoclass:: flowprog.Object
    :members:

.. autoclass:: flowprog.ModelStructure
    :members:
```


## Model building

The idea is to

1. Start with Sympy symbols representing the unknown parameters of the model (e.g. demand of a set of products).
2. Use methods such as {py:meth}`ModelBuilder.pull_production` to find a set of updated equations for various system variables.
3. Optionally, modify these updated equations using methods such as {py:meth}`ModelBuilder.limit`.
4. Add the updated equations into the model using {py:meth}`ModelBuilder.add`, meaning they will now be taken into account for later calculations.

When a complete model has been built, a snapshot is taken and returned as the {py:class}`Model` returned by {py:meth}`ModelBuilder.build`.  From this, expressions for the flows can be exported using {py:meth}`Model.to_flows`. These can either be saved for later use still containing placeholder symbols (e.g. to be filled in later depending on user-specified scenarios or parameters) or can be substituted for specific values directly.  For repeated evaluation the model can be compiled using {py:meth}`Model.lambdify`.

```{eval-rst}
.. autoclass:: flowprog.ModelBuilder
    :members:
```


## Model instances

```{eval-rst}
.. autoclass:: flowprog.Model
    :members:
```
