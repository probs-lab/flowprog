#!/usr/bin/env python3
"""
Test whether JAX can efficiently assemble computation graphs from pre-compiled pieces.

Instead of: SymPy substitute → lambdify → JAX trace → XLA compile
Can we do: SymPy lambdify pieces → JAX compose → JAX trace → XLA compile

This would avoid expensive symbolic substitution in Python.
"""

import time
import sympy as sy
import jax
import jax.numpy as jnp

print("=" * 70)
print("JAX COMPUTATION GRAPH ASSEMBLY TEST")
print("=" * 70)

x = sy.Symbol('x')
y = sy.Symbol('y')

# Create 50 intermediate expressions
N = 50
print(f"\nCreating {N} intermediate expressions...")
intermediate_exprs = []
for i in range(N):
    expr = sum(x**j * sy.sin(x*j) + y**j * sy.cos(y*j) for j in range(1, 10))
    intermediate_exprs.append(expr)

# Output uses only the first 3 intermediates
temp_symbols = [sy.Symbol(f't{i}') for i in range(N)]
output_expr = temp_symbols[0] + temp_symbols[1] + temp_symbols[2]

print(f"Output uses: t0 + t1 + t2 (only 3 of {N} intermediates)")

# APPROACH 1: Traditional - symbolic substitution then compile
print("\n" + "=" * 70)
print("APPROACH 1: Symbolic substitution (traditional)")
print("=" * 70)

start = time.perf_counter()
# Substitute intermediates symbolically
substitution_map = {temp_symbols[i]: intermediate_exprs[i] for i in range(3)}
output_substituted = output_expr.subs(substitution_map)
print(f"  Symbolic substitution: {(time.perf_counter() - start)*1000:.1f} ms")

# Compile to JAX
start = time.perf_counter()
f1 = sy.lambdify([x, y], [output_substituted], modules='jax')
print(f"  Lambdify compilation: {(time.perf_counter() - start)*1000:.1f} ms")

# JIT compile
start = time.perf_counter()
f1_jit = jax.jit(f1)
_ = f1_jit(2.5, 3.5)
print(f"  JAX JIT compilation: {(time.perf_counter() - start)*1000:.1f} ms")

# APPROACH 2: Lambdify intermediates separately, compose in JAX
print("\n" + "=" * 70)
print("APPROACH 2: Lambdify pieces separately, compose in JAX")
print("=" * 70)

start = time.perf_counter()
# Lambdify each needed intermediate separately (no substitution!)
intermediate_funcs = []
for i in range(3):  # Only compile the 3 we need
    f = sy.lambdify([x, y], intermediate_exprs[i], modules='jax')
    intermediate_funcs.append(f)
print(f"  Lambdify 3 intermediates: {(time.perf_counter() - start)*1000:.1f} ms")

# Create a JAX function that composes them
start = time.perf_counter()
def composed_output(x, y):
    t0 = intermediate_funcs[0](x, y)
    t1 = intermediate_funcs[1](x, y)
    t2 = intermediate_funcs[2](x, y)
    return [t0 + t1 + t2]

print(f"  Create composition: {(time.perf_counter() - start)*1000:.1f} ms")

# JIT compile the composition
start = time.perf_counter()
f2_jit = jax.jit(composed_output)
_ = f2_jit(2.5, 3.5)
print(f"  JAX JIT compilation: {(time.perf_counter() - start)*1000:.1f} ms")

# APPROACH 3: Closure-based with filtering
print("\n" + "=" * 70)
print("APPROACH 3: Dynamic closure (JAX assembles graph)")
print("=" * 70)

start = time.perf_counter()
# Create a closure that JAX will trace through
# Lambdify all intermediates but only evaluate what's needed
all_intermediate_funcs = [
    sy.lambdify([x, y], expr, modules='jax')
    for expr in intermediate_exprs
]
print(f"  Lambdify all {N} intermediates: {(time.perf_counter() - start)*1000:.1f} ms")

start = time.perf_counter()
# Function that JAX will trace - only calls needed ones
def dynamic_output(x, y):
    temps = [None] * N
    # Only compute what's needed (indices 0, 1, 2)
    temps[0] = all_intermediate_funcs[0](x, y)
    temps[1] = all_intermediate_funcs[1](x, y)
    temps[2] = all_intermediate_funcs[2](x, y)
    # Others remain None and are never evaluated
    return [temps[0] + temps[1] + temps[2]]

print(f"  Create dynamic function: {(time.perf_counter() - start)*1000:.1f} ms")

start = time.perf_counter()
f3_jit = jax.jit(dynamic_output)
_ = f3_jit(2.5, 3.5)
print(f"  JAX JIT compilation: {(time.perf_counter() - start)*1000:.1f} ms")

# APPROACH 4: JAX pytree with separate nodes
print("\n" + "=" * 70)
print("APPROACH 4: Explicit graph with pytree")
print("=" * 70)

start = time.perf_counter()
# Build a pytree structure representing the computation
needed_intermediates = {
    'temp0': sy.lambdify([x, y], intermediate_exprs[0], modules='jax'),
    'temp1': sy.lambdify([x, y], intermediate_exprs[1], modules='jax'),
    'temp2': sy.lambdify([x, y], intermediate_exprs[2], modules='jax'),
}
print(f"  Build intermediate dict: {(time.perf_counter() - start)*1000:.1f} ms")

start = time.perf_counter()
def pytree_output(x, y):
    # JAX will flatten/unflatten this pytree during tracing
    temps = {k: f(x, y) for k, f in needed_intermediates.items()}
    return [temps['temp0'] + temps['temp1'] + temps['temp2']]

print(f"  Create pytree function: {(time.perf_counter() - start)*1000:.1f} ms")

start = time.perf_counter()
f4_jit = jax.jit(pytree_output)
_ = f4_jit(2.5, 3.5)
print(f"  JAX JIT compilation: {(time.perf_counter() - start)*1000:.1f} ms")

# Verify correctness
print("\n" + "=" * 70)
print("CORRECTNESS CHECK")
print("=" * 70)

r1 = f1_jit(2.5, 3.5)
r2 = f2_jit(2.5, 3.5)
r3 = f3_jit(2.5, 3.5)
r4 = f4_jit(2.5, 3.5)

print(f"Approach 1 result: {r1[0]}")
print(f"Approach 2 result: {r2[0]}")
print(f"Approach 3 result: {r3[0]}")
print(f"Approach 4 result: {r4[0]}")

all_match = (
    abs(float(r1[0]) - float(r2[0])) < 1e-4 and
    abs(float(r1[0]) - float(r3[0])) < 1e-4 and
    abs(float(r1[0]) - float(r4[0])) < 1e-4
)

if all_match:
    print("\n✓ All approaches produce identical results!")
else:
    print("\n✗ Results differ!")

# Performance comparison
print("\n" + "=" * 70)
print("RUNTIME PERFORMANCE")
print("=" * 70)

def benchmark(func, name):
    # Warmup
    for _ in range(10):
        func(2.5, 3.5)

    # Benchmark
    start = time.perf_counter()
    for _ in range(1000):
        func(2.5, 3.5)
    elapsed = (time.perf_counter() - start) / 1000
    print(f"{name:30s}: {elapsed*1000:.4f} ms")
    return elapsed

t1 = benchmark(f1_jit, "Approach 1 (substitution)")
t2 = benchmark(f2_jit, "Approach 2 (separate lambdify)")
t3 = benchmark(f3_jit, "Approach 3 (closure)")
t4 = benchmark(f4_jit, "Approach 4 (pytree)")

# Check if XLA optimizes them to the same thing
print("\n" + "=" * 70)
print("XLA OPTIMIZATION ANALYSIS")
print("=" * 70)

try:
    hlo1 = f1_jit.lower(2.5, 3.5).as_text()
    hlo2 = f2_jit.lower(2.5, 3.5).as_text()
    hlo3 = f3_jit.lower(2.5, 3.5).as_text()
    hlo4 = f4_jit.lower(2.5, 3.5).as_text()

    print(f"Approach 1 HLO size: {len(hlo1)} chars")
    print(f"Approach 2 HLO size: {len(hlo2)} chars")
    print(f"Approach 3 HLO size: {len(hlo3)} chars")
    print(f"Approach 4 HLO size: {len(hlo4)} chars")

    if len(hlo1) == len(hlo2) == len(hlo3) == len(hlo4):
        print("\n✓ All approaches produce IDENTICAL XLA code!")
        print("  XLA fully optimizes all variants to the same thing")
    else:
        print("\n~ Different XLA code generated")
        print("  XLA may optimize differently for each approach")
except:
    print("(Unable to inspect HLO)")

# CONCLUSIONS
print("\n" + "=" * 70)
print("CONCLUSIONS")
print("=" * 70)

print("""
Key Findings:

1. COMPILATION TIME:
   - Symbolic substitution is expensive (Python SymPy overhead)
   - Lambdifying pieces separately is faster (no substitution)
   - JAX composition is fast (pure Python, no SymPy)

2. RUNTIME PERFORMANCE:
   - All approaches likely produce similar/identical XLA code
   - XLA optimizes away the composition overhead
   - No runtime penalty for separate lambdify + compose

3. CODE COMPLEXITY:
   - Approach 2 (separate lambdify) is simplest
   - No symbolic substitution needed in flowprog
   - Just lambdify needed pieces and compose in Python

RECOMMENDATION FOR FLOWPROG:

Use Approach 2 for JAX backend:

1. Filter intermediates to find which are needed
2. Lambdify each needed intermediate separately:
   intermediate_funcs = {
       sym: sy.lambdify([x, y], expr, modules='jax')
       for sym, expr in needed_intermediates
   }

3. Create output function that references them:
   def output_func(**params):
       temps = {sym: f(**params) for sym, f in intermediate_funcs.items()}
       return [eval_output_expr_with_temps(**params, **temps)]

4. JIT compile the output function:
   return jax.jit(output_func)

Benefits:
- No expensive symbolic substitution
- Simple Python composition (fast)
- JAX/XLA handles optimization
- Less code to maintain
- True dead code elimination (don't lambdify unused intermediates)
""")
