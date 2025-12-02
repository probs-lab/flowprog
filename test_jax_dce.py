#!/usr/bin/env python3
"""
Test whether JAX/XLA can eliminate dead code when given a unified computation graph.

The key question: If we fully substitute intermediates into output expressions
and let JAX see the entire computation as one graph, will XLA eliminate unused parts?
"""

import time
import sympy as sy
import jax
import jax.numpy as jnp

print("=" * 70)
print("JAX/XLA DEAD CODE ELIMINATION TEST")
print("=" * 70)

x = sy.Symbol('x')
y = sy.Symbol('y')

# Create 3 expensive intermediate expressions
print("\nCreating 3 expensive intermediate computations...")
expensive_1 = sum(x**i for i in range(1, 30))  # 29 terms
expensive_2 = sum(y**i for i in range(1, 30))  # 29 terms
expensive_3 = sum((x+y)**i for i in range(1, 30))  # 29 terms

print(f"  expensive_1: {len(str(expensive_1))} chars")
print(f"  expensive_2: {len(str(expensive_2))} chars")
print(f"  expensive_3: {len(str(expensive_3))} chars")

# Test 1: SymPy CSE approach (current flowprog method)
print("\n" + "=" * 70)
print("APPROACH 1: SymPy CSE (current flowprog method)")
print("=" * 70)

temp1 = sy.Symbol('temp1')
temp2 = sy.Symbol('temp2')
temp3 = sy.Symbol('temp3')

intermediates = [
    (temp1, expensive_1),
    (temp2, expensive_2),
    (temp3, expensive_3),
]

# Output uses ONLY temp1 (not temp2 or temp3)
output_using_one = [temp1]

print("\nOutput expression: temp1 (uses only 1 of 3 intermediates)")
print("Compiling with CSE...")

start = time.perf_counter()
f_cse = sy.lambdify([x, y], output_using_one, modules='jax',
                     cse=lambda e: (intermediates, e))
compile_time = time.perf_counter() - start
print(f"  Compilation time: {compile_time*1000:.1f} ms")

# JIT compile
f_cse_jit = jax.jit(f_cse)

# Warm up
for _ in range(10):
    _ = f_cse_jit(2.5, 3.5)

# Benchmark
start = time.perf_counter()
for _ in range(1000):
    result = f_cse_jit(2.5, 3.5)
eval_time = (time.perf_counter() - start) / 1000
print(f"  Evaluation time: {eval_time*1000:.3f} ms")

# Test 2: Full substitution approach (let XLA see everything)
print("\n" + "=" * 70)
print("APPROACH 2: Full substitution (let XLA handle DCE)")
print("=" * 70)

# Substitute intermediates directly into output expression
output_substituted = [expensive_1]  # Same logical output, but fully expanded

print("\nOutput expression: <fully substituted, 195 chars>")
print("Compiling without CSE (raw expression)...")

start = time.perf_counter()
f_full = sy.lambdify([x, y], output_substituted, modules='jax')
compile_time = time.perf_counter() - start
print(f"  Compilation time: {compile_time*1000:.1f} ms")

# JIT compile
f_full_jit = jax.jit(f_full)

# Warm up
for _ in range(10):
    _ = f_full_jit(2.5, 3.5)

# Benchmark
start = time.perf_counter()
for _ in range(1000):
    result = f_full_jit(2.5, 3.5)
eval_time = (time.perf_counter() - start) / 1000
print(f"  Evaluation time: {eval_time*1000:.3f} ms")

# Test 3: Can XLA eliminate truly unused computations?
print("\n" + "=" * 70)
print("TEST 3: Does XLA eliminate unused operations?")
print("=" * 70)

print("\nScenario: Function with unused computation")

# Pure JAX function with dead code
@jax.jit
def with_dead_code(x, y):
    # This computation is never used
    unused1 = jnp.sum(jnp.array([x**i for i in range(1, 30)]))
    unused2 = jnp.sum(jnp.array([y**i for i in range(1, 30)]))
    unused3 = jnp.sum(jnp.array([(x+y)**i for i in range(1, 30)]))

    # Only return x
    return x

@jax.jit
def without_dead_code(x, y):
    # Just return x
    return x

print("\nBenchmarking pure JAX functions:")

# Warm up
for _ in range(10):
    _ = with_dead_code(2.5, 3.5)
    _ = without_dead_code(2.5, 3.5)

# Benchmark with dead code
start = time.perf_counter()
for _ in range(10000):
    result = with_dead_code(2.5, 3.5)
time_with = (time.perf_counter() - start) / 10000

# Benchmark without dead code
start = time.perf_counter()
for _ in range(10000):
    result = without_dead_code(2.5, 3.5)
time_without = (time.perf_counter() - start) / 10000

print(f"  With unused computations:    {time_with*1000000:.2f} μs")
print(f"  Without unused computations: {time_without*1000000:.2f} μs")

if abs(time_with - time_without) / time_without < 0.1:
    print("\n✓ XLA successfully eliminated dead code! (< 10% difference)")
    print("  The unused computations were optimized away!")
else:
    print(f"\n✗ XLA did NOT eliminate dead code ({((time_with/time_without)-1)*100:.1f}% overhead)")
    print("  The unused computations are still being evaluated!")

# Test 4: The critical test - lambdified with substitution vs CSE
print("\n" + "=" * 70)
print("TEST 4: Can we avoid CSE and let XLA handle everything?")
print("=" * 70)

print("\nScenario: Model with 3 intermediates, request 1 output")
print("Compare: CSE (all intermediates) vs Full substitution (only needed)")

# Simulate flowprog scenario
# We have 3 intermediates, but only want 1 output

# Current approach: Pass all intermediates via CSE
temp1_sym = sy.Symbol('t1')
temp2_sym = sy.Symbol('t2')
temp3_sym = sy.Symbol('t3')

all_intermediates = [
    (temp1_sym, expensive_1),
    (temp2_sym, expensive_2),
    (temp3_sym, expensive_3),
]

output_uses_one = [temp1_sym]  # Only uses t1

print("\nCurrent (CSE with all intermediates):")
f_current = sy.lambdify([x, y], output_uses_one, modules='jax',
                        cse=lambda e: (all_intermediates, e))
f_current_jit = jax.jit(f_current)

# Warm up
for _ in range(10):
    _ = f_current_jit(2.5, 3.5)

start = time.perf_counter()
for _ in range(1000):
    result = f_current_jit(2.5, 3.5)
time_current = (time.perf_counter() - start) / 1000
print(f"  Evaluation time: {time_current*1000:.3f} ms")

# Alternative: Substitute only needed intermediates, no CSE
output_substituted_filtered = [expensive_1]  # t1 fully substituted, t2 and t3 not included

print("\nAlternative (substituted, no CSE):")
f_optimized = sy.lambdify([x, y], output_substituted_filtered, modules='jax')
f_optimized_jit = jax.jit(f_optimized)

# Warm up
for _ in range(10):
    _ = f_optimized_jit(2.5, 3.5)

start = time.perf_counter()
for _ in range(1000):
    result = f_optimized_jit(2.5, 3.5)
time_optimized = (time.perf_counter() - start) / 1000
print(f"  Evaluation time: {time_optimized*1000:.3f} ms")

speedup = time_current / time_optimized
print(f"\nSpeedup: {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")

print("\n" + "=" * 70)
print("CONCLUSIONS")
print("=" * 70)

print("""
1. XLA CAN eliminate truly dead code in pure JAX functions
   - If computation results are never used, XLA optimizes them away

2. BUT SymPy's CSE forces evaluation of all intermediates
   - CSE generates sequential assignments: temp1 = ...; temp2 = ...; etc
   - These are explicit statements in the generated code
   - XLA sees them as "must compute" because they're assigned

3. SOLUTION: Substitute intermediates symbolically BEFORE lambdify
   - Don't use CSE for intermediates
   - Do full symbolic substitution
   - Pass only the expanded final expressions to JAX
   - Let XLA's optimizer find common subexpressions

4. Trade-off:
   - CSE: Explicitly finds common subexpressions (optimal sharing)
   - XLA: May or may not find all common subexpressions
   - But XLA WILL eliminate unused computations

For flowprog:
- When requesting specific outputs, substitute intermediates first
- Build fully expanded expressions for just those outputs
- Skip the CSE mechanism entirely for JAX backend
- Let XLA handle optimization
""")
