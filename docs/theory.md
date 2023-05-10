# Theory

The structure of the model is consistent with the [PRObs Ontology](https://probs-lab.github.io/probs-ontology), that is:

- The main elements are **Processes**, which represent some transformation, storage, or transportation activities occuring within the model. A Process has an input of some Object(s) and an output of some Object(s).
- An **Object** is a type of thing, including goods, materials and substances, but also non-material things such as energy and services.

The inputs and outputs of a process (its "recipe") can be written as:

$$ \begin{align}
U_{ij} &= \text{use of object $i$ for a unit operation of process $j$} \\
S_{ij} &= \text{supply of object $i$ from a unit operation of process $j$}
\end{align} $$

If the magnitude of output of process $j$ is $Y_j$, then the actual output of material $i$ is scaled up to be $s_{ij} = S_{ij} Y_j$. Similarly, if the magnitude of input into process $j$ is $X_j$, then the actual input of material $i$ is scaled up to be $u_{ij} = U_{ij} X_j$.

In many processes, no accumulation of material happens: the flow in and out is equal. In that case, $X_j = Y_j$. But more generally, there is a *stock accumulation* in the process of $\Delta_j = X_j - Y_j$.

## Mass balance of objects

Conservation of mass should apply for the intermediate object types for which a balancing "market" process is needed:

$$
\sum_j s_{ij} = \sum_j u_{ij}
$$

Conversely, other objects can be treated as "external": they can be consumed and produced by processes freely, and their consumption and production do not need to be balanced within the model system boundary.

## "Pulling" and "pushing" flows

To define the equations making up the model, usually there is some part of the system where we wish to start by specifying the flows using known data, or user-controlled parameters. Then, the model should propagate these specified flows through the system, by:

- Where multiple processes can produce the same object, specifying the relative fractions of demand which should be allocated to each process. For a total demand $D_i$, supply from each possible process $j$ will be:

  $$
  s_{ij} = \alpha_{ij} D_i
  $$
  where $\sum_j \alpha_{ij} = 1$.

- Where the supply from a process $s_{ij}$, has been determined, propagate this to the total output magnitude of the process:

  $$
  Y_j = \frac{s_{ij}}{S_{ij}}
  $$
  
- Where the process has a known (possibly zero) stock accumulation $\Delta_j$, find the total input magnitude of the process:

  $$ X_j = Y_j + \Delta_j $$

  and hence the required use of objects into the process

  $$
  u_{ij} = U_{ij} X_j
  $$
  
- Where the objects should have a balancing market in the model, these new use requirements lead to new demand for supply of the objects:

  $$
  D_i = \sum_j u_{ij}
  $$
  
- These newly-determined further demands for new objects, which can further propagate by repeating these steps (while taking care to deal with the possibility of loops).

Left unrestricted, this propagation will pass through the model until objects are reached which are external inputs (no processes are defined which produce them). Sometimes this is not desired. For example, in a typical material flow model, a stock model will determine demand for new material as well as availability of end-of-life material to be recycled. At some point in the model, there will be a balancing point where supply of recycled material is balanced against demand for new material, with the shortfall being made up from primary production. So, when the demand for new material is "pulled" through the downstream stages of the model, it should reach only as far upstream as this balancing point.

The above steps are written from the perspective of "pulling" demand through the model to cause upstream production, but the opposite applies similarly where supply is "pushed" through the model to cause downstream consumption.

## Implementation using `flowprog`

`flowprog` essentially provides three things:

- Book-keeping for the current values of system variables ($X_j, Y_j$) and the history of modelling steps that contributed to them.
- Functions to compute a full set of flows, caused by propagating supply/demand for an object in one part of the model a series of processes and "markets".
- Functions to query the current values, e.g. to find out how much additional production of an object is required to balance its market.

By computing partial sets of flows, perhaps based on user-specified values or data, and on querying the current interim state of the model, a full model can be built up in simple steps. The following sections describe in greater deal the specific functions that allow you to do this.

Because `flowprog` works with symbolic expressions (equations), the model can be *defined* and tested as a separate step, and then *evaluated* as many times as needed, e.g. to test different parameter values, fit to different countries' data, or to evaluate uncertainty using Monte Carlo sampling.
