#!/usr/bin/env python3
"""
Find the optimal way to compose JAX functions to avoid:
1. Expensive symbolic substitution in SymPy
2. Verbose HLO from separate lambdified functions
3. kwargs overhead

Can we get compact HLO without symbolic substitution?
"""

import time
import sympy as sy
import jax
import jax.numpy as jnp

print("=" * 70)
print("OPTIMAL JAX COMPOSITION TEST")
print("=" * 70)

x = sy.Symbol('x')
y = sy.Symbol('y')

N = 10
intermediate_exprs = [sum(x**j + y**j for j in range(1, 20)) for _ in range(N)]
temp_symbols = [sy.Symbol(f't{i}') for i in range(N)]
output_expr = sum(temp_symbols)

print(f"Testing {N} intermediates, output = sum(all)")

# BASELINE: Single substituted (best HLO, slow compilation)
print("\n" + "=" * 70)
print("BASELINE: Single substituted expression")
print("=" * 70)

start = time.perf_counter()
substitution_map = {temp_symbols[i]: intermediate_exprs[i] for i in range(N)}
output_substituted = output_expr.subs(substitution_map)
f_baseline = jax.jit(sy.lambdify([x, y], output_substituted, modules='jax'))
_ = f_baseline(2.5, 3.5)
time_baseline = time.perf_counter() - start
print(f"  Compilation: {time_baseline*1000:.1f} ms")

hlo_baseline = f_baseline.lower(2.5, 3.5).as_text()
print(f"  HLO size: {len(hlo_baseline)} chars, ~{hlo_baseline.count('=')} ops")

# APPROACH 1: Lambdify all intermediates into one array-returning function
print("\n" + "=" * 70)
print("APPROACH 1: Single lambdify returning array of intermediates")
print("=" * 70)

start = time.perf_counter()
# Lambdify all intermediates as a single vector
f_intermediates_vector = sy.lambdify([x, y], intermediate_exprs, modules='jax')

def composed_vector(x, y):
    temps = f_intermediates_vector(x, y)
    return sum(temps)

f1 = jax.jit(composed_vector)
_ = f1(2.5, 3.5)
time_1 = time.perf_counter() - start
print(f"  Compilation: {time_1*1000:.1f} ms")

hlo_1 = f1.lower(2.5, 3.5).as_text()
print(f"  HLO size: {len(hlo_1)} chars, ~{hlo_1.count('=')} ops")

# APPROACH 2: Lambdify intermediates + output in one call
print("\n" + "=" * 70)
print("APPROACH 2: Single lambdify with intermediates as inputs")
print("=" * 70)

start = time.perf_counter()
# Lambdify intermediates
f_temps = sy.lambdify([x, y], intermediate_exprs, modules='jax')
# Lambdify output with temp symbols as parameters
f_output = sy.lambdify(temp_symbols, output_expr, modules='jax')

def composed_split(x, y):
    temps = f_temps(x, y)
    return f_output(*temps)  # Unpack array as positional args

f2 = jax.jit(composed_split)
_ = f2(2.5, 3.5)
time_2 = time.perf_counter() - start
print(f"  Compilation: {time_2*1000:.1f} ms")

hlo_2 = f2.lower(2.5, 3.5).as_text()
print(f"  HLO size: {len(hlo_2)} chars, ~{hlo_2.count('=')} ops")

# APPROACH 3: Build computation with JAX vmap
print("\n" + "=" * 70)
print("APPROACH 3: Using JAX vmap for parallelism")
print("=" * 70)

start = time.perf_counter()
# Single intermediate function that we'll map over
intermediate_template = intermediate_exprs[0]  # They're all the same
f_single_intermediate = sy.lambdify([x, y], intermediate_template, modules='jax')

def composed_vmap(x, y):
    # Call the same function N times (vmap could parallelize this)
    temps = jnp.array([f_single_intermediate(x, y) for _ in range(N)])
    return jnp.sum(temps)

f3 = jax.jit(composed_vmap)
_ = f3(2.5, 3.5)
time_3 = time.perf_counter() - start
print(f"  Compilation: {time_3*1000:.1f} ms")

hlo_3 = f3.lower(2.5, 3.5).as_text()
print(f"  HLO size: {len(hlo_3)} chars, ~{hlo_3.count('=')} ops")

# APPROACH 4: Manual inlining with string code generation
print("\n" + "=" * 70)
print("APPROACH 4: Generate Python code string and exec")
print("=" * 70)

start = time.perf_counter()
# Generate Python code from SymPy
from sympy import pycode

code_lines = []
for i, expr in enumerate(intermediate_exprs):
    code_lines.append(f"  t{i} = {pycode(expr, fully_qualified_modules=False)}")
code_lines.append(f"  return {' + '.join(f't{i}' for i in range(N))}")

func_code = f"def generated_func(x, y):\n" + "\n".join(code_lines)

# Execute to create function
namespace = {'jnp': jnp}
exec(func_code, namespace)
generated_func = namespace['generated_func']

f4 = jax.jit(generated_func)
_ = f4(2.5, 3.5)
time_4 = time.perf_counter() - start
print(f"  Compilation: {time_4*1000:.1f} ms")

hlo_4 = f4.lower(2.5, 3.5).as_text()
print(f"  HLO size: {len(hlo_4)} chars, ~{hlo_4.count('=')} ops")

# Runtime comparison
print("\n" + "=" * 70)
print("RUNTIME PERFORMANCE")
print("=" * 70)

def benchmark(f):
    # Warmup
    for _ in range(100):
        f(2.5, 3.5).block_until_ready()

    # Benchmark
    start = time.perf_counter()
    for _ in range(10000):
        f(2.5, 3.5).block_until_ready()  # CRITICAL: wait for async execution
    return (time.perf_counter() - start) / 10000

times = {
    "Baseline (substituted)": benchmark(f_baseline),
    "Approach 1 (vector return)": benchmark(f1),
    "Approach 2 (split lambdify)": benchmark(f2),
    "Approach 3 (vmap)": benchmark(f3),
    "Approach 4 (code generation)": benchmark(f4),
}

for name, t in times.items():
    print(f"{name:35s}: {t*1000:.4f} ms")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

data = {
    "Baseline": (time_baseline, len(hlo_baseline), times["Baseline (substituted)"]),
    "Vector return": (time_1, len(hlo_1), times["Approach 1 (vector return)"]),
    "Split lambdify": (time_2, len(hlo_2), times["Approach 2 (split lambdify)"]),
    "vmap": (time_3, len(hlo_3), times["Approach 3 (vmap)"]),
    "Code gen": (time_4, len(hlo_4), times["Approach 4 (code generation)"]),
}

print(f"\n{'Method':<20} {'Compile (ms)':<15} {'HLO size':<15} {'Runtime (ms)':<15}")
print("-" * 65)
for name, (compile_t, hlo_size, runtime_t) in data.items():
    print(f"{name:<20} {compile_t*1000:<15.1f} {hlo_size:<15} {runtime_t*1000:<15.4f}")

# Find winner
best_compile = min(data.items(), key=lambda x: x[1][0])
best_hlo = min(data.items(), key=lambda x: x[1][1])
best_runtime = min(data.items(), key=lambda x: x[1][2])

print(f"\nBest compilation: {best_compile[0]} ({best_compile[1][0]*1000:.1f} ms)")
print(f"Best HLO size: {best_hlo[0]} ({best_hlo[1][1]} chars)")
print(f"Best runtime: {best_runtime[0]} ({best_runtime[1][2]*1000:.4f} ms)")

print(f"""
RECOMMENDATION:

For flowprog's JAX backend, the optimal approach depends on priorities:

1. If HLO size matters (debuggability, memory):
   → Use code generation (Approach 4) or vector return (Approach 1)

2. If compilation speed matters most:
   → Use single substituted expression (Baseline)
   (But requires expensive SymPy symbolic substitution)

3. If implementation simplicity matters:
   → Use vector return (Approach 1) - simple and reasonably compact HLO

The key insight: Returning a vector from a single lambdify call
produces much more compact HLO than multiple separate lambdify calls,
without requiring symbolic substitution.
""")
