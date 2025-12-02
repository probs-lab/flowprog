#!/usr/bin/env python3
"""
Extreme test: 100 expensive intermediates, use only 1
This should clearly show whether XLA can eliminate unused intermediates.
"""

import time
import sympy as sy
import jax
import jax.numpy as jnp

print("=" * 70)
print("EXTREME JAX DCE TEST: 100 intermediates, use 1")
print("=" * 70)

x = sy.Symbol('x')
y = sy.Symbol('y')

# Create 100 expensive intermediate expressions
print("\nCreating 100 expensive intermediate expressions...")
N = 100
intermediates_exprs = []
for i in range(N):
    # Each is moderately expensive
    expr = sum(x**j * sy.sin(x*j) + y**j * sy.cos(y*j) for j in range(1, 10))
    intermediates_exprs.append(expr)

print(f"  Created {N} intermediates, each with ~18 terms")

# Test 1: CSE with ALL intermediates (current approach)
print("\n" + "=" * 70)
print("TEST 1: SymPy CSE with ALL intermediates")
print("=" * 70)

intermediate_symbols = [sy.Symbol(f't{i}') for i in range(N)]
cse_list = [(sym, expr) for sym, expr in zip(intermediate_symbols, intermediates_exprs)]

# Output uses ONLY the first intermediate
output_cse = [intermediate_symbols[0]]

print(f"\nOutput uses: t0 (only 1 of {N} intermediates)")
print("Compiling with CSE...")

start = time.perf_counter()
f_cse = sy.lambdify([x, y], output_cse, modules='jax',
                     cse=lambda e: (cse_list, e))
compile_time = time.perf_counter() - start
print(f"  Compilation time: {compile_time*1000:.1f} ms")

# JIT compile
print("  Applying JIT...")
start = time.perf_counter()
f_cse_jit = jax.jit(f_cse)
# Warm up (triggers JIT compilation)
_ = f_cse_jit(2.5, 3.5)
jit_time = time.perf_counter() - start
print(f"  JIT compilation time: {jit_time*1000:.1f} ms")

# Benchmark
print("  Benchmarking...")
for _ in range(10):  # More warmup
    _ = f_cse_jit(2.5, 3.5)

start = time.perf_counter()
for _ in range(1000):
    result = f_cse_jit(2.5, 3.5)
eval_time = (time.perf_counter() - start) / 1000
print(f"  Evaluation time: {eval_time*1000:.3f} ms")

# Test 2: Full substitution - ONLY needed intermediate
print("\n" + "=" * 70)
print("TEST 2: Full substitution - only needed intermediate")
print("=" * 70)

# Substitute: just use the first expression directly
output_substituted = [intermediates_exprs[0]]

print(f"\nOutput: <first intermediate fully expanded>")
print("Compiling without CSE...")

start = time.perf_counter()
f_sub = sy.lambdify([x, y], output_substituted, modules='jax')
compile_time = time.perf_counter() - start
print(f"  Compilation time: {compile_time*1000:.1f} ms")

# JIT compile
print("  Applying JIT...")
start = time.perf_counter()
f_sub_jit = jax.jit(f_sub)
_ = f_sub_jit(2.5, 3.5)
jit_time = time.perf_counter() - start
print(f"  JIT compilation time: {jit_time*1000:.1f} ms")

# Benchmark
print("  Benchmarking...")
for _ in range(10):
    _ = f_sub_jit(2.5, 3.5)

start = time.perf_counter()
for _ in range(1000):
    result = f_sub_jit(2.5, 3.5)
eval_time_sub = (time.perf_counter() - start) / 1000
print(f"  Evaluation time: {eval_time_sub*1000:.3f} ms")

speedup = eval_time / eval_time_sub
print(f"\n{'='*70}")
print(f"SPEEDUP: {speedup:.2f}x by skipping unused intermediates")
print(f"{'='*70}")

if speedup > 1.5:
    print("\n✓ SIGNIFICANT BENEFIT from avoiding CSE!")
    print(f"  CSE forces evaluation of all {N} intermediates")
    print("  Substitution + XLA only computes what's needed")
elif speedup > 1.1:
    print("\n✓ Moderate benefit from avoiding CSE")
    print("  Some overhead from unused intermediates")
else:
    print("\n~ Similar performance")
    print("  XLA may be optimizing away the unused intermediates")
    print("  Or expressions are too simple to show difference")

# Test 3: Verify correctness
print("\n" + "=" * 70)
print("TEST 3: Verify correctness")
print("=" * 70)

result_cse = f_cse_jit(2.5, 3.5)
result_sub = f_sub_jit(2.5, 3.5)

print(f"CSE result:          {result_cse}")
print(f"Substitution result: {result_sub}")
print(f"Difference:          {abs(float(result_cse[0]) - float(result_sub[0])):.2e}")

if abs(float(result_cse[0]) - float(result_sub[0])) < 1e-6:
    print("\n✓ Results match!")
else:
    print("\n✗ Results differ!")

# Test 4: Inspect what XLA compiled
print("\n" + "=" * 70)
print("TEST 4: Inspect XLA compilation")
print("=" * 70)

print("\nCSE approach HLO size:")
try:
    hlo_cse = f_cse_jit.lower(2.5, 3.5).as_text()
    print(f"  HLO text length: {len(hlo_cse)} chars")
    print(f"  HLO instruction count: ~{hlo_cse.count('=')}")
except:
    print("  (Unable to inspect HLO)")

print("\nSubstitution approach HLO size:")
try:
    hlo_sub = f_sub_jit.lower(2.5, 3.5).as_text()
    print(f"  HLO text length: {len(hlo_sub)} chars")
    print(f"  HLO instruction count: ~{hlo_sub.count('=')}")
except:
    print("  (Unable to inspect HLO)")

print("\n" + "=" * 70)
print("RECOMMENDATION FOR FLOWPROG")
print("=" * 70)

print(f"""
When compiling with JAX backend and requesting specific expressions:

1. Don't use CSE for intermediates
2. Symbolically substitute intermediates into outputs FIRST
3. Only substitute intermediates that are actually needed
4. Pass expanded expressions to sy.lambdify(..., modules='jax')
5. Let XLA handle optimization

Implementation:
  def _lambdify_jax_optimized(self, values, data_for_intermediates):
      # Find which intermediates are needed
      needed_symbols = find_dependencies(values, self._intermediates)

      # Substitute ONLY needed intermediates
      for sym, expr, _ in self._intermediates:
          if sym in needed_symbols:
              values = [v.subs(sym, expr) for v in values]

      # Substitute data
      values = [v.xreplace(data_for_intermediates) for v in values]

      # Compile WITHOUT CSE - let XLA handle it
      f = sy.lambdify(args, values, modules='jax')
      return jax.jit(f)

This gives you TRUE dead code elimination at the symbolic level,
and XLA can still find common subexpressions within the compiled code.
""")
