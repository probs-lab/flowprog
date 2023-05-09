# Model API

The building blocks for a model are the `Process` and `Object` definitions:

```{eval-rst}
.. autoclass:: flowprog.imperative_model.Process
    :members:

.. autoclass:: flowprog.imperative_model.Object
    :members:
```

Then, the `Model` class defines a structure of a model consisting of `Process`es and `Object`s.

The idea is to

1. Start with Sympy symbols representing the unknown parameters of the model (e.g. demand of a set of products).
2. Use methods such as {py:method}`Model.pull_production` to find a set of updated equations for various system variables.
3. Optionally, modify these updated equations using methods such as {py:method}`Model.limit`.
4. Add the updated equations into the model using {py:method}`Model.add`, meaning they will now be taken into account for later calculations.

When a complete model has been built, expressions for the flows can be exported using {py:method}`Model.to_flows`. These can either be saved for later use still containing placeholder symbols (e.g. to be filled in later depending on user-specified scenarios or parameters) or can be substituted for specific values directly.

```{eval-rst}
.. autoclass:: flowprog.imperative_model.Model
    :members:
```

