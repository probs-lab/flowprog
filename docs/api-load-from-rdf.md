# Loading from RDF API

To create a {py:class}`flowprog.imperative_model.Model` from RDF data, use `query_model_from_endpoint`.

This function expects an object with a `query` method, such as an RDFlib graph or a `rdfox_runner` `RDFoxEndpoint`, to query for Processes and Objects. It also loads recipe data in `sphinx_probs_rdf` format.

```{eval-rst}
.. autofunction:: flowprog.load_from_rdf.query_model_from_endpoint

```
