#!/usr/bin/env python3
"""
Demonstrate dependency tracking algorithm to find needed intermediates.
"""

import sympy as sy

print("=" * 70)
print("DEPENDENCY TRACKING ALGORITHM")
print("=" * 70)

# Setup: Create a chain of intermediates
x = sy.Symbol('x')
y = sy.Symbol('y')

# Intermediates form a dependency chain
t0 = sy.Symbol('t0')
t1 = sy.Symbol('t1')
t2 = sy.Symbol('t2')
t3 = sy.Symbol('t3')
t4 = sy.Symbol('t4')
t5 = sy.Symbol('t5')

intermediates = [
    (t0, x + y),           # Independent
    (t1, x * y),           # Independent
    (t2, t0 + t1),         # Depends on t0, t1
    (t3, t2 * 2),          # Depends on t2 (which depends on t0, t1)
    (t4, x**2),            # Independent
    (t5, t3 + t4),         # Depends on t3, t4 (transitively on t0, t1, t2)
]

print("\nIntermediate expressions:")
for sym, expr in intermediates:
    print(f"  {sym} = {expr}")

# User requests only t5
output_expressions = [t5 + 10]

print(f"\nRequested output: {output_expressions[0]}")

# ALGORITHM: Find all needed intermediates
print("\n" + "=" * 70)
print("DEPENDENCY TRACKING")
print("=" * 70)

def find_needed_intermediates(output_exprs, intermediates):
    """
    Find which intermediates are needed to evaluate the output expressions.

    Args:
        output_exprs: List of sympy expressions we want to evaluate
        intermediates: List of (symbol, expression) tuples

    Returns:
        Set of intermediate symbols that are needed
    """
    # Build lookup: symbol -> expression
    intermediate_map = {sym: expr for sym, expr in intermediates}

    # Start with symbols used in outputs
    needed_symbols = set()
    for expr in output_exprs:
        needed_symbols.update(expr.free_symbols)

    print(f"Step 1: Symbols in output expressions: {needed_symbols}")

    # Iteratively find dependencies
    # Keep expanding until no new symbols found
    iteration = 1
    while True:
        new_symbols = set()

        for sym in needed_symbols:
            if sym in intermediate_map:
                # This is an intermediate - find what it depends on
                intermediate_expr = intermediate_map[sym]
                deps = intermediate_expr.free_symbols
                new_deps = deps - needed_symbols
                if new_deps:
                    print(f"  Iteration {iteration}: {sym} depends on {new_deps}")
                    new_symbols.update(new_deps)

        if not new_symbols:
            break  # No new dependencies found

        needed_symbols.update(new_symbols)
        iteration += 1

    # Filter to only intermediate symbols (not input variables like x, y)
    intermediate_symbols = {sym for sym, _ in intermediates}
    needed_intermediates = needed_symbols & intermediate_symbols

    print(f"\nFinal needed intermediates: {needed_intermediates}")
    print(f"Out of {len(intermediates)} total intermediates")

    return needed_intermediates

needed = find_needed_intermediates(output_expressions, intermediates)

# Show what we'd compile
print("\n" + "=" * 70)
print("COMPILATION PLAN")
print("=" * 70)

print("\nWould compile these intermediates:")
for sym, expr in intermediates:
    if sym in needed:
        print(f"  ✓ {sym} = {expr}")
    else:
        print(f"  ✗ {sym} = {expr}  (SKIPPED)")

print(f"\nSavings: {len(intermediates) - len(needed)}/{len(intermediates)} intermediates skipped")

# Example 2: More complex case
print("\n" + "=" * 70)
print("EXAMPLE 2: Multiple outputs with shared dependencies")
print("=" * 70)

output_exprs_2 = [t2, t4]  # Request t2 and t4
print(f"Requested outputs: {output_exprs_2}")

needed_2 = find_needed_intermediates(output_exprs_2, intermediates)

print("\nWould compile:")
for sym, expr in intermediates:
    if sym in needed_2:
        print(f"  ✓ {sym} = {expr}")
    else:
        print(f"  ✗ {sym} = {expr}  (SKIPPED)")

# Example 3: Real flowprog scenario
print("\n" + "=" * 70)
print("EXAMPLE 3: Flowprog-like scenario")
print("=" * 70)

# Simulate flowprog intermediates
S = sy.IndexedBase('S')
U = sy.IndexedBase('U')
d = sy.Symbol('d')

# Intermediate expressions (like flowprog's _intermediates)
flowprog_intermediates = [
    (sy.Symbol('Y0'), d / S[0, 0]),                    # Process 0 output
    (sy.Symbol('X0'), sy.Symbol('Y0') * U[0, 0]),      # Process 0 input
    (sy.Symbol('Y1'), sy.Symbol('X0') / S[1, 1]),      # Process 1 output (depends on X0)
    (sy.Symbol('Y2'), d / S[2, 2]),                    # Process 2 output (independent)
    (sy.Symbol('deficit'), sy.Max(0, d - sy.Symbol('Y2'))),  # Deficit calculation
]

print("Intermediates:")
for sym, expr in flowprog_intermediates:
    print(f"  {sym} = {expr}")

# User requests only Y1 (which creates a dependency chain Y1 -> X0 -> Y0)
output_flowprog = [sy.Symbol('Y1')]
print(f"\nRequested: {output_flowprog[0]}")

needed_flowprog = find_needed_intermediates(output_flowprog, flowprog_intermediates)

print("\nCompilation plan:")
for sym, expr in flowprog_intermediates:
    if sym in needed_flowprog:
        print(f"  ✓ Compile {sym}")
    else:
        print(f"  ✗ Skip {sym}")

print("\n" + "=" * 70)
print("IMPLEMENTATION FOR FLOWPROG")
print("=" * 70)

print("""
In flowprog, add this method:

def _find_needed_intermediates(self, output_expressions):
    '''Find which intermediates are needed for given outputs.'''

    # Build lookup
    intermediate_map = {sym: expr for sym, expr, _ in self._intermediates}

    # Start with output dependencies
    needed = set()
    for expr in output_expressions:
        needed.update(expr.free_symbols)

    # Expand dependencies iteratively
    while True:
        new_deps = set()
        for sym in needed:
            if sym in intermediate_map:
                new_deps.update(intermediate_map[sym].free_symbols)

        new_deps -= needed  # Only new ones
        if not new_deps:
            break
        needed.update(new_deps)

    # Filter to intermediate symbols only
    intermediate_symbols = {sym for sym, _, _ in self._intermediates}
    return needed & intermediate_symbols

Then use it in _lambdify_jax:

def _lambdify_jax(self, values, data_for_intermediates):
    # Find which intermediates we actually need
    needed_symbols = self._find_needed_intermediates(values)

    # Only process needed ones
    filtered_intermediates = [
        (sym, expr, desc)
        for sym, expr, desc in self._intermediates
        if sym in needed_symbols
    ]

    # Continue with only filtered_intermediates...
""")
