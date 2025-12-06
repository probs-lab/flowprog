#!/usr/bin/env python3
"""Test whether providing namespace to sympify actually uses the provided objects."""

import sympy as sy

print("Testing whether sympify uses provided namespace for IndexedBase\n")

# Create original IndexedBase
S_original = sy.IndexedBase('S', shape=(3, 2))
expr = S_original[0, 1] + S_original[1, 0]
print(f"Original expression: {expr}")
print(f"ID of S_original: {id(S_original)}")
print(f"ID of S in expr[0]: {id(expr.args[0].base)}")
print(f"ID of S in expr[1]: {id(expr.args[1].base)}")
print(f"Are they all the same object? {S_original is expr.args[0].base is expr.args[1].base}")
print()

# Serialize
expr_str = sy.srepr(expr)
print(f"Serialized (srepr): {expr_str[:80]}...")
print()

# Test 1: Deserialize WITHOUT namespace
print("="*70)
print("Test 1: Deserialize WITHOUT providing namespace")
print("="*70)
recreated1 = sy.sympify(expr_str)
S_recreated1_ref1 = recreated1.args[0].base
S_recreated1_ref2 = recreated1.args[1].base
print(f"ID of S_original:     {id(S_original)}")
print(f"ID of S in recreated[0]: {id(S_recreated1_ref1)}")
print(f"ID of S in recreated[1]: {id(S_recreated1_ref2)}")
print(f"All references in recreated share same base? {S_recreated1_ref1 is S_recreated1_ref2}")
print(f"Recreated base is original? {S_recreated1_ref1 is S_original}")
print(f"Result: Sympify creates {'SAME' if S_recreated1_ref1 is S_original else 'NEW'} IndexedBase object")
print()

# Test 2: Deserialize WITH namespace
print("="*70)
print("Test 2: Deserialize WITH providing namespace")
print("="*70)
S_provided = sy.IndexedBase('S', shape=(3, 2))
print(f"Created S_provided with ID: {id(S_provided)}")
recreated2 = sy.sympify(expr_str, locals={'S': S_provided})
S_recreated2_ref1 = recreated2.args[0].base
S_recreated2_ref2 = recreated2.args[1].base
print(f"ID of S_provided:        {id(S_provided)}")
print(f"ID of S in recreated[0]:    {id(S_recreated2_ref1)}")
print(f"ID of S in recreated[1]:    {id(S_recreated2_ref2)}")
print(f"All references in recreated share same base? {S_recreated2_ref1 is S_recreated2_ref2}")
print(f"Recreated base is S_provided? {S_recreated2_ref1 is S_provided}")
print(f"Result: Sympify {'USES PROVIDED' if S_recreated2_ref1 is S_provided else 'IGNORES PROVIDED'} IndexedBase")
print()

# Test 3: Does it matter?
print("="*70)
print("Test 3: Does it matter for practical use?")
print("="*70)
print("Checking if both recreated expressions are functionally equivalent:")
print(f"  recreated1 == recreated2: {recreated1 == recreated2}")
print(f"  recreated1 free_symbols: {recreated1.free_symbols}")
print(f"  recreated2 free_symbols: {recreated2.free_symbols}")
print()

# Test with lambdify
print("Testing with lambdify:")
import numpy as np
test_data = {S_original[0, 1]: 10, S_original[1, 0]: 20}
try:
    func1 = sy.lambdify([S_recreated1_ref1], recreated1)
    result1 = func1(np.array([[0, 10], [20, 0], [0, 0]]))
    print(f"  lambdify with recreated1 (no namespace): {result1}")
except Exception as e:
    print(f"  lambdify with recreated1: ERROR - {e}")

try:
    func2 = sy.lambdify([S_provided], recreated2)
    result2 = func2(np.array([[0, 10], [20, 0], [0, 0]]))
    print(f"  lambdify with recreated2 (with namespace): {result2}")
except Exception as e:
    print(f"  lambdify with recreated2: ERROR - {e}")

print()
print("="*70)
print("CONCLUSION:")
print("="*70)
print("""
The srepr() output contains COMPLETE information about IndexedBase including:
- Name
- Shape
- Assumptions (like nonnegative=True)

When sympify recreates an IndexedBase from srepr:
- It creates a NEW Python object (different id)
- But it's functionally EQUIVALENT (same name, shape, assumptions)
- All references within one sympify call share the SAME new object
- Providing namespace dict does NOT override this behavior

For flowprog serialization, this means:
1. You DON'T need to track user-defined IndexedBase objects separately
2. srepr/sympify will handle them automatically
3. As long as you deserialize the whole model at once, references will be consistent
4. Lambdify will work correctly with the recreated objects

The only thing to track: Model's X, Y, S, U should be created once during __init__
and then all expressions loaded from the serialized format will reference the
newly created IndexedBase objects from their srepr strings.
""")
