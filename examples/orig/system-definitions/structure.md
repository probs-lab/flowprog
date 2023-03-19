---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
--- 

Primary chemicals production
============================

```{system:process} PrimaryChemicalsProduction
:label: Primary chemicals production
:consumes: RefineryProducts NaturalGas
:produces: PrimaryChemicals NaturalGas
:become_parent: true
```

```{system:process} HydrogenSynthesis
:label: Hydrogen synthesis
:consumes: FossilFuels
:produces: Hydrogen
:become_parent: true
```

```{system:process} HydrogenSynthesisFromNaturalGas
---
label: Hydrogen synthesis from natural gas
consumes: |
  ukf:NaturalGas                               = 1.0 kg
  ukf:Water                                    = 1.0 kg
produces: |
  ukf:CO2                                      = 1.0 kg
  ukf:Hydrogen                                 = 1.0 kg
---

TODO: made up recipe

Steam methane reforming followed by water gas shift reaction.

```

```{system:process} HydrogenSynthesisFromCoal
---
label: Hydrogen synthesis from coal
consumes: |
  ukf:Coal                                     = 1.0 kg
  ukf:Water                                    = 1.0 kg
produces: |
  ukf:CO2                                      = 1.0 kg
  ukf:Hydrogen                                 = 1.0 kg
---

TODO: made up recipe

Partial oxidisation followed by water gas shift reaction.

```

```{end-sub-processes}
```

```{system:process} AmmoniaSynthesis
---
consumes: |
  ukf:Hydrogen                                 = 0.5 kg
  ukf:Nitrogen                                 = 0.5 kg
produces: |
  ukf:Ammonia                                  = 1.0 kg
---

TODO: made up recipe
The IHS database probably contains data on ammonia production in the UK. These data should be added here.

This process is defined slightly differently from UK FIRES -- consuming Hydrogen
and Nitrogen, not Syngas.
```

```{end-sub-processes}
```

### Misc objects

```{system:object} CO2

CO2 in a pure stream that can be re-used in other processes, as opposed to `AtmosphericCO2`.
```

Fertilliser production
======================

````{system:process} ProducingUrea
---
consumes: |
  ukf:Ammonia                                  = 0.58 kg
  CO2                                      = 0.75 kg {comment: 'check CO2 type: atmospheric CO2 or other CO2'}
produces: |
  ukf:Urea                                     = 1.0 kg
  ukf:Water                                    = 0.30 kg
  ukf:OtherIndustrialGases                     = 0.03 kg {comment: 'CO2'}
---

Copied from UK FIRES (unchanged).
````

````{system:process} ProducingAmmoniumNitrate
---
consumes: |
  ukf:Ammonia                                  = 0.22 kg
  ukf:NitricAcid                               = 0.80 kg {comment: 'nitric acid(HNO3)'}
produces: |
  ukf:AmmoniumNitrate                          = 1.0 kg
  ukf:WasteOtherChemicals                      = 0.02 kg {comment: 'nitric acid(HNO3)'}
---
Copied from UK FIRES (unchanged).
````

```{system:process} ProducingNitricAcid
---
consumes: |
  ukf:Ammonia                                  = 0.34 kg
  ukf:PureOxygen                               = 1.28 kg
  ukf:Water                                    = 0.18 kg
produces: |
  ukf:NitricAcid                               = 1.26 kg
  ukf:Water                                    = 0.54 kg
---

Copied from UK FIRES -- but the recipe is not balanced FIXME

FIXME doesn't include N2O emissions.
```

# Use of fertilisers

```{system:process} UseOfUreaFertiliser
---
consumes: |
  ukf:Urea                                     = 1.00 kg
produces: |
  ukf:AtmosphericCO2                           = 1.00 kg
---

FIXME: illustrative recipe!
```

## Balancing processes

```{system:process} ExtractCO2FromAtmosphere
---
consumes: |
  ukf:AtmosphericCO2                           = 1.00 kg
produces: |
  CO2                                      = 1.00 kg
---
```
