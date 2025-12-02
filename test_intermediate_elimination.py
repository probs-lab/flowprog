#!/usr/bin/env python3
"""
Test whether unused intermediates are evaluated in flowprog's lambdify.

This tests the actual Model.lambdify behavior with CSE intermediates.
"""

import time
import sympy as sy
from flowprog.imperative_model import Model, Process, Object
from rdflib import URIRef

MASS = URIRef("http://qudt.org/vocab/quantitykind/Mass")

print("=" * 70)
print("INTERMEDIATE EVALUATION TEST")
print("=" * 70)

# Create a model
processes = [Process(f"P{i}", produces=[f"O{i}"], consumes=[f"O{i-1}"])
             for i in range(1, 101)]  # 100 processes in a chain
processes[0] = Process("P1", produces=["O1"], consumes=["O0"])

objects = [Object(f"O{i}", MASS) for i in range(101)]

model = Model(processes, objects)

# Create parameters
params = {sy.Symbol(f'd{i}'): i * 10.0 for i in range(100)}

# Add flows - each one creates intermediate expressions
print("\nBuilding model with 100 chained processes...")
for i in range(100):
    demand = sy.Symbol(f'd{i}')
    # Pull production creates intermediates
    result = model.pull_production(f"O{i+1}", demand, until_objects=[f"O{i}"])
    model.add(result, label=f"Flow {i}")

print(f"  Processes: {len(model.processes)}")
print(f"  Objects: {len(model.objects)}")
print(f"  Intermediates: {len(model._intermediates)}")

# Recipe data
recipe_data = {}
for j in range(len(model.processes)):
    for i in range(len(model.objects)):
        if model.S[i, j] in model._values or i in [j, j+1]:  # Only set relevant ones
            recipe_data[model.S[i, j]] = 1.0
            recipe_data[model.U[i, j]] = 1.0

print("\n" + "=" * 70)
print("TEST 1: Compile ALL flows")
print("=" * 70)

start = time.perf_counter()
func_all = model.lambdify(data=recipe_data, backend='numpy')
compile_time = time.perf_counter() - start
print(f"Compilation time: {compile_time*1000:.1f} ms")

# Evaluate
all_params = {**params}
start = time.perf_counter()
result_all = func_all(all_params)
eval_time = time.perf_counter() - start
print(f"Evaluation time: {eval_time*1000:.2f} ms")
print(f"Number of flows: {len(result_all)}")

print("\n" + "=" * 70)
print("TEST 2: Compile SPECIFIC flow expressions")
print("=" * 70)

# Get specific flow expressions (just the first 5)
flows_sym = model.to_flows(recipe_data, flow_ids=True)
specific_flows = {
    flows_sym.iloc[i]['id']: flows_sym.iloc[i]['value']
    for i in range(min(5, len(flows_sym)))
}

print(f"Requesting {len(specific_flows)} specific flows (out of {len(flows_sym)})")

start = time.perf_counter()
func_specific = model.lambdify(data=recipe_data, expressions=specific_flows, backend='numpy')
compile_time = time.perf_counter() - start
print(f"Compilation time: {compile_time*1000:.1f} ms")

# Evaluate
start = time.perf_counter()
result_specific = func_specific(all_params)
eval_time = time.perf_counter() - start
print(f"Evaluation time: {eval_time*1000:.2f} ms")
print(f"Number of flows: {len(result_specific)}")

print("\n" + "=" * 70)
print("TEST 3: Compile a SINGLE simple expression")
print("=" * 70)

# Create a trivial expression that doesn't use intermediates
simple_expr = {
    'simple': sy.Symbol('d0')  # Just returns input parameter
}

print(f"Expression: d0 (trivial, no intermediates needed)")

start = time.perf_counter()
func_simple = model.lambdify(data=recipe_data, expressions=simple_expr, backend='numpy')
compile_time = time.perf_counter() - start
print(f"Compilation time: {compile_time*1000:.1f} ms")

# Evaluate
start = time.perf_counter()
for _ in range(100):
    result_simple = func_simple(all_params)
eval_time = (time.perf_counter() - start) / 100
print(f"Evaluation time (avg of 100): {eval_time*1000:.3f} ms")
print(f"Result: {result_simple}")

print("\n" + "=" * 70)
print("ANALYSIS")
print("=" * 70)

print(f"""
Current Behavior:
- Model has {len(model._intermediates)} intermediate expressions
- When requesting specific outputs, expressions are filtered ✓
- But ALL {len(model._intermediates)} intermediates are still compiled ✗

This means:
- Even trivial expressions carry intermediate computation overhead
- Could optimize by filtering intermediates based on dependencies

Potential Optimization:
1. Track which intermediates each expression uses
2. Only include needed intermediates in subexpressions list
3. This is symbolic DCE at the intermediate level
""")

# Try to inspect the generated function
import inspect
print("\nGenerated function for simple expression:")
print(f"  Code object: {func_simple.__code__.co_name}")
print(f"  # of local variables: {func_simple.__code__.co_nlocals}")
print(f"  Variable names: {func_simple.__code__.co_varnames[:10]}...")  # First 10
if func_simple.__code__.co_nlocals > 50:
    print(f"\n  ⚠️  Function has {func_simple.__code__.co_nlocals} local variables")
    print(f"      This includes ALL {len(model._intermediates)} intermediates!")
    print(f"      Even though the output is just 'd0'")
