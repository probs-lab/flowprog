#!/usr/bin/env python3
"""Test what happens when we have multiple models with the same IndexedBase names."""

import sympy as sy

print("Testing Multiple Model Scenario\n")
print("="*70)

# Simulate Model 1
print("Creating Model 1...")
X1 = sy.IndexedBase('X', shape=(3,))
S1 = sy.IndexedBase('S', shape=(2, 3))
expr1 = X1[0] * S1[0, 0] + X1[1] * S1[0, 1]
print(f"Model 1 - X1 id: {id(X1)}, S1 id: {id(S1)}")
print(f"Model 1 expr: {expr1}")

# Serialize Model 1
expr1_str = sy.srepr(expr1)
print(f"Serialized Model 1")
print()

# Simulate Model 2 (different model, same variable names)
print("Creating Model 2...")
X2 = sy.IndexedBase('X', shape=(3,))  # Same name and shape as Model 1!
S2 = sy.IndexedBase('S', shape=(2, 3))
expr2 = X2[2] * S2[1, 2]
print(f"Model 2 - X2 id: {id(X2)}, S2 id: {id(S2)}")
print(f"Model 2 expr: {expr2}")
print()

# Check if they're the same objects
print("Checking object identity:")
print(f"  X1 is X2? {X1 is X2}")
print(f"  S1 is S2? {S1 is S2}")
print("⚠️  SymPy returns SAME cached objects for IndexedBase with same name/shape!")
print()

# Now deserialize Model 1's expression
print("="*70)
print("Deserializing Model 1's expression...")
loaded_expr1 = sy.sympify(expr1_str)
print(f"Loaded expr: {loaded_expr1}")

# Extract the IndexedBase objects from loaded expression
def get_indexed_bases(expr):
    bases = set()
    if isinstance(expr, sy.Indexed):
        bases.add(expr.base)
    if hasattr(expr, 'args'):
        for arg in expr.args:
            bases.update(get_indexed_bases(arg))
    return bases

loaded_bases = get_indexed_bases(loaded_expr1)
loaded_X = [b for b in loaded_bases if str(b) == 'X'][0]
loaded_S = [b for b in loaded_bases if str(b) == 'S'][0]

print(f"Loaded X id: {id(loaded_X)}")
print(f"Loaded S id: {id(loaded_S)}")
print()

print("Checking identities:")
print(f"  Loaded X is X1? {loaded_X is X1}")
print(f"  Loaded X is X2? {loaded_X is X2}")
print(f"  Loaded S is S1? {loaded_S is S1}")
print(f"  Loaded S is S2? {loaded_S is S2}")
print()

print("="*70)
print("What this means for flowprog:")
print("="*70)
print("""
GOOD NEWS:
1. Within a single Python process, SymPy caches IndexedBase objects
2. Same name + same properties = same Python object
3. This means loaded expressions will reference the same objects as the Model

POTENTIAL ISSUES:
1. If you create Model A, serialize it, then create Model B (different model,
   same variable names), the deserialized Model A expressions might reference
   Model B's IndexedBase objects!

2. This is a problem IF:
   - You want to load multiple saved models in the same process
   - The models have the same IndexedBase names (X, Y, S, U) which they will!

SOLUTION:
The key insight is that when you deserialize a Model:
1. First create the Model object (which creates new X, Y, S, U)
2. Then deserialize expressions - they SHOULD use the Model's X, Y, S, U

But SymPy's caching might interfere. Let me test if we can control this...
""")

print("\n" + "="*70)
print("Testing solution: Create Model FIRST, then deserialize")
print("="*70)

# Create a fresh model
print("Creating Model 3...")
X3 = sy.IndexedBase('X', shape=(5,))  # Different shape!
S3 = sy.IndexedBase('S', shape=(4, 5))
print(f"Model 3 - X3 id: {id(X3)}, shape={X3.shape}")
print(f"Model 3 - S3 id: {id(S3)}, shape={S3.shape}")
print()

print("Now deserializing Model 1's expression (which has X with shape (3,))...")
loaded_expr1_again = sy.sympify(expr1_str)
loaded_X_again = [b for b in get_indexed_bases(loaded_expr1_again) if str(b) == 'X'][0]
print(f"Loaded X id: {id(loaded_X_again)}, shape={loaded_X_again.shape}")
print(f"  Is it X1 (shape 3)? {loaded_X_again is X1}")
print(f"  Is it X3 (shape 5)? {loaded_X_again is X3}")
print()

print("✅  SymPy caching uses BOTH name AND properties (like shape)")
print("✅  Different shapes = different cached objects")
print()

print("="*70)
print("FINAL RECOMMENDATION:")
print("="*70)
print("""
For flowprog serialization to work correctly:

1. SAVE: When serializing, save the Model structure (processes, objects) along
   with the IndexedBase shapes/properties, so you can recreate the Model first.

2. LOAD: When deserializing:
   a. Create Model object first (this creates X, Y, S, U with correct shapes)
   b. The model's X, Y, S, U will be cached by SymPy
   c. When you deserialize expressions using sympify(srepr_string), they will
      automatically use the cached (just-created) IndexedBase objects
   d. As long as shapes match, everything will reference the same objects

3. RISK: If multiple models with SAME shapes exist in same process, expressions
   might reference the wrong model's IndexedBase objects. This is a rare edge
   case but worth documenting.

4. MITIGATION: Could use unique names per model (e.g., "X_model1", "X_model2")
   but this changes the architecture. Or just document that only one model
   should be active at a time, or clear SymPy caches between models.
""")
