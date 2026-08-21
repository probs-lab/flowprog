# System definitions

The `{system:process}` and `{system:object}` blocks below define the processes and objects that make up the system.  This format of definition is equivalent to writing Python code listing the `Process()` and `Object()` declarations directly, but this way is convenient as

- it allows definitions to be grouped into logical sections
- it gives space to document the definitions and sources of recipe assumptions
- it can be formatted automatically into a nice human-readable HTML or PDF documentation file

## EoL technology

```{system:process} EoLProcessing
---
consumes: |
produces: |
  EoLTechnology                            = 1 kg
---

This represents the dMFA model with flows leaving in-use stock
```

```{system:object} EoLTechnology
```

```{system:process} Disassembly
---
consumes: |
  EoLTechnology                            = 1 kg
produces: |
  PCBs                                     = 0.2 kg
  OtherParts                               = 0.8 kg
---

```

```{system:object} PCBs
```

```{system:object} OtherParts
```

## PCB recycling

```{system:process} PCBProcess1
---
consumes: |
  PCBs                                     = 1 kg
produces: |
  PureGold                                 = 0.1 kg
  MixedPCBWaste                            = 0.9 kg
---
```

```{system:process} PCBProcess2
---
consumes: |
  PCBs                                     = 1 kg
produces: |
  GoldAndPlastic                           = 0.3 kg
  MixedPCBWaste                            = 0.7 kg
---
```

```{system:process} GoldAndPlasticProcessing
---
consumes: |
  GoldAndPlastic                           = 1 kg
produces: |
  PureGold                                 = 0.2 kg
  MixedPCBWaste                            = 0.8 kg
---
```

```{system:object} PureGold
```

```{system:object} GoldAndPlastic
```

```{system:object} MixedPCBWaste
You can explain more about the objects too here.
```
