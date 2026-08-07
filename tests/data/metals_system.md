# Metals system

A small system used to check that `flowprog.io.system_definitions` and
`flowprog.io.rdf` load the same model. `metals_system.ttl` is the RDF form of these
same definitions, as exported by a Sphinx build.

## Extraction

```{system:process} Extraction
:consumes: Gas
:produces: Ore Metal
:become_parent: true
```

```{system:process} Mining
---
consumes: |
produces: |
  Ore                                      = 1 kg
---
```

```{system:process} Smelting
---
consumes: |
  Ore                                      = 2 kg
  Electricity                              = 3 kWh
produces: |
  Metal                                    = 1 kg
---
```

```{end-sub-processes}
```

## Power

```{system:process} PowerStation
---
consumes: |
  Gas                                      = 0.2 kg
produces: |
  Electricity                              = 1 kWh
---
```

## Object definitions

```{system:object} Ore
```

```{system:object} Metal
```

```{system:object} Electricity
```

```{system:object} Gas
```
