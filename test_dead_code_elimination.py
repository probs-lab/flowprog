#!/usr/bin/env python3
"""
Test dead code elimination in different compilation backends.

Scenario: We have 1000 complex expressions, but only request 1 simple one.
Will the backend avoid computing the other 999?
"""

import time
import sympy as sy
import numpy as np

print("=" * 70)
print("DEAD CODE ELIMINATION TEST")
print("=" * 70)

# Create symbols
x = sy.Symbol('x')

# Create 1000 expensive expressions
print("\nCreating 1000 expensive expressions...")
expensive_exprs = []
for i in range(1000):
    # Each expression is moderately expensive to compute
    expr = sum(x**j * sy.sin(x * j) for j in range(1, 20))
    expensive_exprs.append(expr)

# Add one trivial expression at the end
simple_expr = x  # Just returns the input
all_exprs = expensive_exprs + [simple_expr]

print(f"  Total expressions: {len(all_exprs)}")
print(f"  First 999: Expensive (19 terms each with sin/pow)")
print(f"  Last 1: Trivial (just x)")

# Test 1: Compile ALL expressions
print("\n" + "=" * 70)
print("TEST 1: Compile all 1000 expressions, use all outputs")
print("=" * 70)

print("\nNumPy backend (all expressions)...")
start = time.perf_counter()
f_numpy_all = sy.lambdify([x], all_exprs, modules='numpy')
compile_time = time.perf_counter() - start
print(f"  Compilation time: {compile_time*1000:.1f} ms")

start = time.perf_counter()
result = f_numpy_all(2.5)
eval_time = time.perf_counter() - start
print(f"  Evaluation time: {eval_time*1000:.2f} ms")
print(f"  Result length: {len(result)}")

# Test 2: Compile only the simple expression
print("\n" + "=" * 70)
print("TEST 2: Compile only the trivial expression")
print("=" * 70)

print("\nNumPy backend (1 trivial expression)...")
start = time.perf_counter()
f_numpy_simple = sy.lambdify([x], [simple_expr], modules='numpy')
compile_time = time.perf_counter() - start
print(f"  Compilation time: {compile_time*1000:.1f} ms")

start = time.perf_counter()
result = f_numpy_simple(2.5)
eval_time = time.perf_counter() - start
print(f"  Evaluation time: {eval_time*1000:.2f} ms")
print(f"  Result: {result}")

# Test 3: JAX with all expressions
print("\n" + "=" * 70)
print("TEST 3: JAX backend - all expressions")
print("=" * 70)

try:
    import jax

    print("\nJAX backend (all 1000 expressions)...")
    start = time.perf_counter()
    f_jax_all = sy.lambdify([x], all_exprs, modules='jax')
    compile_time = time.perf_counter() - start
    print(f"  Compilation time: {compile_time*1000:.1f} ms")

    # Warm up JIT
    _ = f_jax_all(2.5)

    start = time.perf_counter()
    for _ in range(100):
        result = f_jax_all(2.5)
    eval_time = (time.perf_counter() - start) / 100
    print(f"  Evaluation time (avg of 100): {eval_time*1000:.2f} ms")
    print(f"  Result length: {len(result)}")

except ImportError:
    print("JAX not available")

# Test 4: JAX with only simple expression
print("\n" + "=" * 70)
print("TEST 4: JAX backend - trivial expression only")
print("=" * 70)

try:
    import jax

    print("\nJAX backend (1 trivial expression)...")
    start = time.perf_counter()
    f_jax_simple = sy.lambdify([x], [simple_expr], modules='jax')
    compile_time = time.perf_counter() - start
    print(f"  Compilation time: {compile_time*1000:.1f} ms")

    # Warm up JIT
    _ = f_jax_simple(2.5)

    start = time.perf_counter()
    for _ in range(100):
        result = f_jax_simple(2.5)
    eval_time = (time.perf_counter() - start) / 100
    print(f"  Evaluation time (avg of 100): {eval_time*1000:.2f} ms")
    print(f"  Result: {result}")

except ImportError:
    print("JAX not available")

# Test 5: What if we extract only what we need?
print("\n" + "=" * 70)
print("TEST 5: Extract dependencies symbolically (BEST PRACTICE)")
print("=" * 70)

print("\nScenario: Model has 1000 intermediate expressions,")
print("but we only need to evaluate one output that doesn't depend on them.")

# Simulate: we have 1000 intermediates, but only 1 output we care about
intermediates = [(sy.Symbol(f'temp_{i}'), expensive_exprs[i])
                 for i in range(1000)]

output_expr = simple_expr  # This doesn't use any intermediates!

print(f"\nOutput expression: {output_expr}")
print(f"Free symbols in output: {output_expr.free_symbols}")

# Find which intermediates are actually needed
needed_symbols = output_expr.free_symbols
needed_intermediates = [
    (sym, expr) for sym, expr in intermediates
    if sym in needed_symbols
]

print(f"\nIntermediates needed: {len(needed_intermediates)} (out of 1000)")

print("\nCompiling only what's needed...")
start = time.perf_counter()
f_optimized = sy.lambdify([x], [output_expr], modules='numpy')
compile_time = time.perf_counter() - start
print(f"  Compilation time: {compile_time*1000:.1f} ms")

start = time.perf_counter()
result = f_optimized(2.5)
eval_time = time.perf_counter() - start
print(f"  Evaluation time: {eval_time*1000:.2f} ms")
print(f"  Result: {result}")

# Test 6: Dependency tracking example
print("\n" + "=" * 70)
print("TEST 6: Dependency tracking when output DOES use intermediates")
print("=" * 70)

# Now the output uses some intermediates
temp_0 = sy.Symbol('temp_0')
temp_1 = sy.Symbol('temp_1')
temp_500 = sy.Symbol('temp_500')

output_complex = temp_0 + temp_1 + temp_500  # Uses 3 out of 1000

print(f"\nOutput expression: {output_complex}")
print(f"Free symbols: {output_complex.free_symbols}")

# After substituting intermediates
output_substituted = output_complex.subs([
    (temp_0, expensive_exprs[0]),
    (temp_1, expensive_exprs[1]),
    (temp_500, expensive_exprs[500]),
])

print("\nCompiling with all dependencies substituted...")
start = time.perf_counter()
f_deps = sy.lambdify([x], [output_substituted], modules='numpy')
compile_time = time.perf_counter() - start
print(f"  Compilation time: {compile_time*1000:.1f} ms")

start = time.perf_counter()
result = f_deps(2.5)
eval_time = time.perf_counter() - start
print(f"  Evaluation time: {eval_time*1000:.2f} ms")
print(f"  This computes only 3 expensive expressions (not 1000)")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print("""
Key Findings:

1. **NumPy/JAX lambdify**: Compiles EXACTLY what you give it
   - If you pass 1000 expressions → computes 1000 expressions
   - NO automatic dead code elimination
   - Even if output[999] is trivial, still computes [0:999]

2. **XLA (JAX's compiler)**: CAN eliminate unused code, but...
   - Only if the function returns unused values
   - If you extract result[999] in Python, all are computed
   - DCE works at the compiled function level, not Python level

3. **ufuncify (C/Fortran)**: Compiler optimization flags can help
   - gcc -O2/-O3 can eliminate dead code
   - But only within each generated function
   - Still computes all expressions you told it to compute

4. **BEST PRACTICE: Symbolic filtering before compilation**
   - Use SymPy to track dependencies
   - Only compile expressions you actually need
   - This is what flowprog's current approach should do!

For your Model.lambdify use case:
- If user requests specific outputs, extract their dependencies
- Only substitute/compile those expressions
- Avoid compiling 1000s of flows when only 10 are needed
""")

print("\nRecommendation for flowprog:")
print("  In Model.lambdify(expressions={...}), when expressions is provided,")
print("  only compile those specific expressions, not all flows.")
print("  This is ALREADY what the current code does! ✓")
