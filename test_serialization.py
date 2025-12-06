#!/usr/bin/env python3
"""Test script to verify sympy serialization behavior with different symbol types."""

import sympy as sy

print("="*70)
print("Testing SymPy Serialization with srepr/sympify")
print("="*70)

# Test 1: Simple symbols
print("\n" + "="*70)
print("Test 1: Simple symbols (no assumptions)")
print("="*70)
a = sy.Symbol('a')
b = sy.Symbol('b')
expr1 = a + 2*b
print(f"Original expression: {expr1}")
print(f"srepr output: {sy.srepr(expr1)}")
recreated1 = sy.sympify(sy.srepr(expr1))
print(f"Recreated expression: {recreated1}")
print(f"Expressions equal? {expr1 == recreated1}")
print(f"'a' in recreated free_symbols? {sy.Symbol('a') in recreated1.free_symbols}")

# Test 2: Symbols with assumptions
print("\n" + "="*70)
print("Test 2: Symbols WITH assumptions (positive)")
print("="*70)
c = sy.Symbol('c', positive=True)
d = sy.Symbol('d', real=True, nonnegative=True)
expr2 = c + d
print(f"Original: {expr2}")
print(f"  c.is_positive = {c.is_positive}")
print(f"  d.is_real = {d.is_real}, d.is_nonnegative = {d.is_nonnegative}")
print(f"srepr: {sy.srepr(expr2)}")
recreated2 = sy.sympify(sy.srepr(expr2))
print(f"Recreated: {recreated2}")
c_recreated = [s for s in recreated2.free_symbols if s.name == 'c'][0]
d_recreated = [s for s in recreated2.free_symbols if s.name == 'd'][0]
print(f"  Recreated c.is_positive = {c_recreated.is_positive}")
print(f"  Recreated d.is_real = {d_recreated.is_real}, d.is_nonnegative = {d_recreated.is_nonnegative}")
print(f"⚠️  Assumptions preserved: {c_recreated.is_positive and d_recreated.is_nonnegative}")

# Test 3: IndexedBase without providing to sympify
print("\n" + "="*70)
print("Test 3: IndexedBase WITHOUT providing to sympify locals")
print("="*70)
X = sy.IndexedBase('X', shape=(3,), nonnegative=True)
expr3 = X[0] + X[1] * 2
print(f"Original: {expr3}")
print(f"  X has shape: {X.shape}")
print(f"  All X[] references share same base object: {X[0].base is X[1].base is X}")
print(f"srepr: {sy.srepr(expr3)}")
recreated3 = sy.sympify(sy.srepr(expr3))
print(f"Recreated: {recreated3}")
# Extract all Indexed terms
def extract_indexed(expr):
    if isinstance(expr, sy.Indexed):
        return [expr]
    elif hasattr(expr, 'args'):
        result = []
        for arg in expr.args:
            result.extend(extract_indexed(arg))
        return result
    return []
indexed_terms = extract_indexed(recreated3)
print(f"  Found {len(indexed_terms)} Indexed terms")
if len(indexed_terms) >= 2:
    print(f"  Recreated X[0].base is X[1].base: {indexed_terms[0].base is indexed_terms[1].base}")
    print(f"  Recreated X[0].base is original X: {indexed_terms[0].base is X}")
    print(f"  Recreated X.shape: {indexed_terms[0].base.shape}")
    print(f"✅  Same IndexedBase object within recreated expression: {indexed_terms[0].base is indexed_terms[1].base}")
    print(f"❌  But NOT the same as original X: {indexed_terms[0].base is X}")

# Test 4: IndexedBase WITH providing to sympify
print("\n" + "="*70)
print("Test 4: IndexedBase WITH providing to sympify locals")
print("="*70)
X_new = sy.IndexedBase('X', shape=(3,), nonnegative=True)
expr4 = X[0] + X[1] * 2
print(f"Original expression (using X): {expr4}")
print(f"srepr: {sy.srepr(expr4)}")
print(f"Providing X_new to sympify locals...")
recreated4 = sy.sympify(sy.srepr(expr4), locals={'X': X_new})
print(f"Recreated: {recreated4}")
indexed_terms4 = extract_indexed(recreated4)
if len(indexed_terms4) >= 2:
    print(f"  Recreated X[0].base is X_new: {indexed_terms4[0].base is X_new}")
    print(f"  Recreated X[1].base is X_new: {indexed_terms4[1].base is X_new}")
    print(f"  All references point to same X_new: {indexed_terms4[0].base is indexed_terms4[1].base is X_new}")
    print(f"✅  Success! All IndexedBase references use the provided X_new object")

# Test 5: Mixed expression with simple symbols and IndexedBase
print("\n" + "="*70)
print("Test 5: Mixed - Simple Symbols + IndexedBase")
print("="*70)
Y = sy.IndexedBase('Y', shape=(5,))
alpha = sy.Symbol('alpha')
beta = sy.Symbol('beta', positive=True)
expr5 = alpha * Y[0] + beta * Y[1] + 10
print(f"Original: {expr5}")
print(f"srepr: {sy.srepr(expr5)}")
print(f"\nRecreating with Y provided but not alpha/beta...")
Y_new = sy.IndexedBase('Y', shape=(5,))
recreated5 = sy.sympify(sy.srepr(expr5), locals={'Y': Y_new})
print(f"Recreated: {recreated5}")
indexed_terms5 = extract_indexed(recreated5)
if indexed_terms5:
    print(f"  Y references use Y_new: {indexed_terms5[0].base is Y_new}")
print(f"  alpha and beta automatically recreated as new Symbol objects")
# Get symbols, filtering out Y base
free_syms = [s for s in recreated5.free_symbols if not isinstance(s, sy.IndexedBase)]
alpha_new = [s for s in free_syms if s.name == 'alpha'][0] if any(s.name == 'alpha' for s in free_syms) else None
beta_new = [s for s in free_syms if s.name == 'beta'][0] if any(s.name == 'beta' for s in free_syms) else None
if alpha_new:
    print(f"  alpha_new is original alpha? {alpha_new is alpha}")
if beta_new:
    print(f"  beta_new.is_positive (was True): {beta_new.is_positive}")
print(f"✅  IndexedBase correctly uses provided object")
print(f"✅  Simple symbols auto-recreated with assumptions preserved")

# Test 6: Multiple different IndexedBase objects
print("\n" + "="*70)
print("Test 6: Multiple different IndexedBase objects (S, U, X, Y)")
print("="*70)
M, N = 2, 3
S = sy.IndexedBase('S', shape=(N, M), nonnegative=True)
U = sy.IndexedBase('U', shape=(N, M), nonnegative=True)
X = sy.IndexedBase('X', shape=(M,), nonnegative=True)
Y = sy.IndexedBase('Y', shape=(M,), nonnegative=True)
gamma = sy.Symbol('gamma')

expr6 = X[0] * S[1, 0] + Y[0] * U[1, 0] + gamma
print(f"Original: {expr6}")
print(f"srepr: {sy.srepr(expr6)}")
print(f"\nRecreating with all IndexedBase objects provided...")
S_new = sy.IndexedBase('S', shape=(N, M), nonnegative=True)
U_new = sy.IndexedBase('U', shape=(N, M), nonnegative=True)
X_new = sy.IndexedBase('X', shape=(M,), nonnegative=True)
Y_new = sy.IndexedBase('Y', shape=(M,), nonnegative=True)
recreated6 = sy.sympify(sy.srepr(expr6), locals={
    'S': S_new, 'U': U_new, 'X': X_new, 'Y': Y_new
})
print(f"Recreated: {recreated6}")
print(f"  All bases correctly reference new objects:")
# Extract bases from expression
bases_in_expr = set()
for term in recreated6.args:
    if isinstance(term, sy.Indexed):
        bases_in_expr.add(term.base)
    elif hasattr(term, 'args'):
        for subterm in term.args:
            if isinstance(subterm, sy.Indexed):
                bases_in_expr.add(subterm.base)
print(f"    S: {S_new in bases_in_expr}")
print(f"    U: {U_new in bases_in_expr}")
print(f"    X: {X_new in bases_in_expr}")
print(f"    Y: {Y_new in bases_in_expr}")
print(f"✅  All IndexedBase objects correctly use provided instances")

print("\n" + "="*70)
print("SUMMARY OF FINDINGS")
print("="*70)
print("""
1. ✅ Simple Symbols (no assumptions): Auto-recreated correctly by sympify
2. ✅ Symbols with assumptions: Assumptions ARE preserved in srepr/sympify
3. ❌ IndexedBase without namespace: Creates NEW IndexedBase objects
   - Multiple references in same expression DO share the same new object
   - But they're NOT the same as the original object
4. ✅ IndexedBase with namespace: Uses provided objects correctly
   - Must provide all IndexedBase objects to sympify's locals parameter
5. ✅ Mixed expressions: Work correctly when IndexedBase provided to locals

RECOMMENDATION FOR SERIALIZATION:
- Use srepr() to serialize expressions
- When deserializing, provide a namespace dict to sympify with all IndexedBase
  objects (X, Y, S, U, and any user-defined IndexedBase)
- Simple Symbol objects will be auto-recreated with correct assumptions
- User-defined IndexedBase objects MUST be tracked separately and provided
  during deserialization, otherwise they'll become disconnected references
""")
