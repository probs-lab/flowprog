#!/usr/bin/env python3
"""
When does JAX actually beat NumPy?

Test different scenarios:
1. Single evaluation
2. Sequential evaluations (many parameter sets)
3. Batch/vectorized evaluation
"""

import time
import numpy as np
import sympy as sy
import jax
import jax.numpy as jnp

print("=" * 70)
print("NUMPY vs JAX: WHEN DOES JAX WIN?")
print("=" * 70)

# Create a moderate complexity expression
x = sy.Symbol('x')
y = sy.Symbol('y')

# 10 intermediates
N = 10
intermediate_exprs = [sum(x**j + y**j for j in range(1, 15)) for _ in range(N)]
temp_symbols = [sy.Symbol(f't{i}') for i in range(N)]
output_expr = sum(temp_symbols[:3])  # Use first 3

# Compile both versions (using the better JAX approach)
print("\nCompiling both backends...")

# NumPy version (filter intermediates)
start = time.perf_counter()
needed_symbols = output_expr.free_symbols & set(temp_symbols)
needed_indices = [i for i in range(N) if temp_symbols[i] in needed_symbols]

# NumPy: Use CSE
numpy_intermediates = [(temp_symbols[i], intermediate_exprs[i]) for i in needed_indices]
f_numpy = sy.lambdify([x, y], [output_expr], modules='numpy',
                      cse=lambda e: (numpy_intermediates, e))
numpy_compile_time = time.perf_counter() - start
print(f"NumPy compilation: {numpy_compile_time*1000:.1f} ms")

# JAX version (composition approach)
start = time.perf_counter()
intermediate_funcs = {
    temp_symbols[i]: sy.lambdify([x, y], intermediate_exprs[i], modules='jax')
    for i in needed_indices
}
output_func = sy.lambdify([x, y] + list(needed_symbols), output_expr, modules='jax')

def jax_composed(x, y):
    temps = {str(sym): func(x, y) for sym, func in intermediate_funcs.items()}
    return output_func(x, y, **temps)

f_jax = jax.jit(jax_composed)
# Warmup
_ = f_jax(2.5, 3.5)
jax_compile_time = time.perf_counter() - start
print(f"JAX compilation (inc JIT): {jax_compile_time*1000:.1f} ms")

# SCENARIO 1: Single evaluation
print("\n" + "=" * 70)
print("SCENARIO 1: Single evaluation (cold start)")
print("=" * 70)

start = time.perf_counter()
r_numpy = f_numpy(2.5, 3.5)
numpy_single = time.perf_counter() - start

start = time.perf_counter()
r_jax = f_jax(2.5, 3.5)
if hasattr(r_jax, 'block_until_ready'):
    r_jax.block_until_ready()
jax_single = time.perf_counter() - start

print(f"NumPy: {numpy_single*1000:.4f} ms")
print(f"JAX:   {jax_single*1000:.4f} ms")
print(f"Winner: {'NumPy' if numpy_single < jax_single else 'JAX'} ({min(numpy_single, jax_single)/max(numpy_single, jax_single):.2%} faster)")

# SCENARIO 2: Sequential evaluations (warmed up)
print("\n" + "=" * 70)
print("SCENARIO 2: Sequential evaluations (1000 parameter sets)")
print("=" * 70)

# Generate 1000 different parameter sets
np.random.seed(42)
param_sets = [(np.random.rand() * 10, np.random.rand() * 10) for _ in range(1000)]

# NumPy - sequential
start = time.perf_counter()
for x_val, y_val in param_sets:
    r = f_numpy(x_val, y_val)
numpy_seq = time.perf_counter() - start

# JAX - sequential (warmed up)
# Warmup first
for _ in range(10):
    f_jax(2.5, 3.5)

start = time.perf_counter()
for x_val, y_val in param_sets:
    r = f_jax(x_val, y_val)
    if hasattr(r, 'block_until_ready'):
        r.block_until_ready()
jax_seq = time.perf_counter() - start

print(f"NumPy: {numpy_seq*1000:.1f} ms total ({numpy_seq/len(param_sets)*1000:.4f} ms/eval)")
print(f"JAX:   {jax_seq*1000:.1f} ms total ({jax_seq/len(param_sets)*1000:.4f} ms/eval)")
print(f"Winner: {'NumPy' if numpy_seq < jax_seq else 'JAX'} ({min(numpy_seq, jax_seq)/max(numpy_seq, jax_seq):.2%} faster)")

# SCENARIO 3: Vectorized/batch evaluation
print("\n" + "=" * 70)
print("SCENARIO 3: Vectorized batch evaluation (1000 at once)")
print("=" * 70)

# NumPy - vectorized
x_vals = np.array([x for x, y in param_sets])
y_vals = np.array([y for x, y in param_sets])

start = time.perf_counter()
r_numpy_vec = f_numpy(x_vals, y_vals)
numpy_vec = time.perf_counter() - start

# JAX - vectorized with vmap
@jax.jit
def jax_single_eval(x, y):
    temps = {str(sym): func(x, y) for sym, func in intermediate_funcs.items()}
    return output_func(x, y, **temps)

jax_vectorized = jax.jit(jax.vmap(jax_single_eval))
# Warmup
_ = jax_vectorized(jnp.array([2.5]), jnp.array([3.5]))

start = time.perf_counter()
r_jax_vec = jax_vectorized(jnp.array(x_vals), jnp.array(y_vals))
r_jax_vec.block_until_ready()
jax_vec = time.perf_counter() - start

print(f"NumPy vectorized: {numpy_vec*1000:.1f} ms")
print(f"JAX vmap:         {jax_vec*1000:.1f} ms")
print(f"Winner: {'NumPy' if numpy_vec < jax_vec else 'JAX'} ({min(numpy_vec, jax_vec)/max(numpy_vec, jax_vec):.2%} faster)")

# SCENARIO 4: Include compilation time
print("\n" + "=" * 70)
print("SCENARIO 4: Total time including compilation")
print("=" * 70)

print(f"\nFor 1 evaluation:")
numpy_total_1 = numpy_compile_time + numpy_single
jax_total_1 = jax_compile_time + jax_single
print(f"  NumPy: {numpy_total_1*1000:.1f} ms")
print(f"  JAX:   {jax_total_1*1000:.1f} ms")
print(f"  Winner: {'NumPy' if numpy_total_1 < jax_total_1 else 'JAX'}")

print(f"\nFor 1000 evaluations (sequential):")
numpy_total_1000 = numpy_compile_time + numpy_seq
jax_total_1000 = jax_compile_time + jax_seq
print(f"  NumPy: {numpy_total_1000*1000:.1f} ms")
print(f"  JAX:   {jax_total_1000*1000:.1f} ms")
print(f"  Winner: {'NumPy' if numpy_total_1000 < jax_total_1000 else 'JAX'}")

# Calculate break-even point
print("\n" + "=" * 70)
print("BREAK-EVEN ANALYSIS")
print("=" * 70)

compile_overhead = jax_compile_time - numpy_compile_time
per_eval_benefit = numpy_seq/1000 - jax_seq/1000

if per_eval_benefit > 0:
    breakeven = compile_overhead / per_eval_benefit
    print(f"\nCompilation overhead: {compile_overhead*1000:.1f} ms")
    print(f"Per-evaluation benefit: {per_eval_benefit*1000:.4f} ms")
    print(f"\nBreak-even point: {breakeven:.0f} evaluations")
    print(f"  - Below {breakeven:.0f} evals: NumPy is faster")
    print(f"  - Above {breakeven:.0f} evals: JAX is faster")
else:
    print(f"\nJAX is slower per evaluation, NumPy always wins for this case")

# Summary
print("\n" + "=" * 70)
print("SUMMARY: WHEN TO USE JAX")
print("=" * 70)

print("""
Based on these tests:

1. SINGLE EVALUATION:
   ✗ JAX is SLOWER (compilation overhead not amortized)
   → Use NumPy for one-off calculations

2. SEQUENTIAL EVALUATIONS (many parameter sets):
   Result: Depends on break-even point
   → If you evaluate the model > ~{} times, JAX wins
   → For interactive analysis with <{} evals, NumPy is better

3. VECTORIZED/BATCH EVALUATION:
   Result: Depends on problem complexity
   → JAX vmap CAN provide speedup for parallel work
   → But for simple expressions, NumPy vectorization is very fast

4. GPU/TPU ACCELERATION:
   ✓ JAX provides access to GPUs/TPUs
   → For very large models or massive batch sizes, JAX can be much faster
   → NumPy is CPU-only

RECOMMENDATION FOR FLOWPROG:

- Default backend: NumPy (simplest, fastest for typical use)
- JAX backend: Optional for users who:
  * Need to evaluate 100s-1000s of parameter sets
  * Want to use GPUs/TPUs
  * Are doing optimization (auto-differentiation)
  * Can tolerate compilation overhead

The JAX backend is a good option to have, but NumPy should remain default.
""".format(int(breakeven) if per_eval_benefit > 0 else "N/A",
          int(breakeven) if per_eval_benefit > 0 else "N/A"))
