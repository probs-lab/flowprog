# Example model structure

The processes and objects are defined in `system-definitions.md`, and read from there
directly -- there is no separate statement of which processes make up the model, or of
which objects need markets. Both follow from the definitions: every process with a
recipe is included, and an object gets a market if the model both produces and
consumes it. Here that means `EoLTechnology`, `PCBs` and `GoldAndPlastic` are
balanced, while `PureGold`, `OtherParts` and `MixedPCBWaste` are only produced, so
they leave the model boundary unbalanced.

The structure of the model is then built up in `load_model.py`:

```{literalinclude} load_model.py
:language: python
```
