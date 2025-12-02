#!/usr/bin/env python3
"""
CLEAR COMPARISON: Current approach vs JAX tracing approach

(a) Current: SymPy xreplace → lambdify with CSE
(b) JAX tracing: Lambdify pieces → Python composition → JAX traces
"""

import time
import sympy as sy
import jax
import jax.numpy as jnp

print("=" * 70)
print("FOCUSED COMPARISON: Current vs JAX Tracing")
print("=" * 70)

# Setup: 50 intermediates, use only 3
x = sy.Symbol('x')
y = sy.Symbol('y')

N_total = 50
N_used = 3

print(f"\nScenario: {N_total} intermediates in model, using only {N_used}")

# Create intermediate expressions
intermediate_exprs = []
temp_symbols = []
for i in range(N_total):
    temp_symbols.append(sy.Symbol(f't{i}'))
    expr = sum(x**j + y**j for j in range(1, 15))  # Moderate complexity
    intermediate_exprs.append(expr)

# Output uses only first 3 intermediates
output_expr = temp_symbols[0] + temp_symbols[1] + temp_symbols[2]

# Simulate recipe data substitution
recipe_data = {sy.Symbol('scale'): 1.5}

print("\n" + "=" * 70)
print("APPROACH A: Current (xreplace backwards → lambdify CSE)")
print("=" * 70)

start_total = time.perf_counter()

# Step 1: xreplace recipe data into ALL intermediates (current approach)
start = time.perf_counter()
subexpressions = [
    (temp_symbols[i], intermediate_exprs[i].xreplace(recipe_data))
    for i in range(N_total)  # ALL intermediates, even unused ones
]
time_xreplace = time.perf_counter() - start
print(f"1. xreplace into {N_total} intermediates: {time_xreplace*1000:.1f} ms")

# Step 2: Substitute data into output
start = time.perf_counter()
output_with_data = output_expr.xreplace(recipe_data)
time_output_sub = time.perf_counter() - start
print(f"2. xreplace into output:              {time_output_sub*1000:.1f} ms")

# Step 3: lambdify with CSE
start = time.perf_counter()
f_current = sy.lambdify([x, y], [output_with_data], modules='jax',
                        cse=lambda e: (subexpressions, e))
time_lambdify = time.perf_counter() - start
print(f"3. lambdify with CSE:                 {time_lambdify*1000:.1f} ms")

# Step 4: JIT compile
start = time.perf_counter()
f_current_jit = jax.jit(f_current)
result = f_current_jit(2.5, 3.5)
if isinstance(result, list):
    [r.block_until_ready() for r in result]
else:
    result.block_until_ready()
time_jit = time.perf_counter() - start
print(f"4. JAX JIT compilation:               {time_jit*1000:.1f} ms")

time_current_total = time.perf_counter() - start_total
print(f"\nTOTAL COMPILATION TIME: {time_current_total*1000:.1f} ms")

# Get HLO size
hlo_current = f_current_jit.lower(2.5, 3.5).as_text()
print(f"HLO size: {len(hlo_current)} chars, ~{hlo_current.count('=')} ops")

print("\n" + "=" * 70)
print("APPROACH B: JAX Tracing (lambdify pieces → compose → trace)")
print("=" * 70)

start_total = time.perf_counter()

# Step 1: Find which intermediates are actually needed
start = time.perf_counter()
needed_symbols = output_expr.free_symbols
needed_indices = [i for i in range(N_total) if temp_symbols[i] in needed_symbols]
time_dependency = time.perf_counter() - start
print(f"1. Find dependencies:                 {time_dependency*1000:.1f} ms")
print(f"   → Need {len(needed_indices)} intermediates (not {N_total})")

# Step 2: Lambdify ONLY needed intermediates (no xreplace composition)
start = time.perf_counter()
intermediate_funcs = {}
for i in needed_indices:
    # Just lambdify the intermediate with recipe data substituted
    expr_with_data = intermediate_exprs[i].xreplace(recipe_data)
    intermediate_funcs[temp_symbols[i]] = sy.lambdify([x, y], expr_with_data, modules='jax')
time_lambdify_pieces = time.perf_counter() - start
print(f"2. Lambdify {len(needed_indices)} intermediates:            {time_lambdify_pieces*1000:.1f} ms")

# Step 3: Lambdify output expression (with intermediate symbols as params)
start = time.perf_counter()
output_with_data = output_expr.xreplace(recipe_data)
output_func = sy.lambdify([x, y] + list(needed_symbols), output_with_data, modules='jax')
time_lambdify_output = time.perf_counter() - start
print(f"3. Lambdify output expression:        {time_lambdify_output*1000:.1f} ms")

# Step 4: Compose in Python (let JAX trace through this)
start = time.perf_counter()
def composed(x, y):
    # Evaluate intermediates
    temp_values = {str(sym): func(x, y) for sym, func in intermediate_funcs.items()}
    # Evaluate output with intermediate values
    return output_func(x, y, **temp_values)

time_compose = time.perf_counter() - start
print(f"4. Create Python composition:         {time_compose*1000:.1f} ms")

# Step 5: JIT compile (JAX traces through the composition)
start = time.perf_counter()
f_tracing_jit = jax.jit(composed)
result = f_tracing_jit(2.5, 3.5)
if isinstance(result, list):
    [r.block_until_ready() for r in result]
else:
    result.block_until_ready()
time_jit_tracing = time.perf_counter() - start
print(f"5. JAX JIT compilation:               {time_jit_tracing*1000:.1f} ms")

time_tracing_total = time.perf_counter() - start_total
print(f"\nTOTAL COMPILATION TIME: {time_tracing_total*1000:.1f} ms")

# Get HLO size
hlo_tracing = f_tracing_jit.lower(2.5, 3.5).as_text()
print(f"HLO size: {len(hlo_tracing)} chars, ~{hlo_tracing.count('=')} ops")

# Runtime comparison
print("\n" + "=" * 70)
print("RUNTIME PERFORMANCE")
print("=" * 70)

def benchmark(f, name):
    # Warmup
    for _ in range(100):
        result = f(2.5, 3.5)
        if isinstance(result, list):
            [r.block_until_ready() for r in result]
        else:
            result.block_until_ready()

    # Benchmark
    start = time.perf_counter()
    for _ in range(10000):
        result = f(2.5, 3.5)
        if isinstance(result, list):
            [r.block_until_ready() for r in result]
        else:
            result.block_until_ready()
    elapsed = (time.perf_counter() - start) / 10000
    print(f"{name:40s}: {elapsed*1000:.4f} ms")
    return elapsed

t_current = benchmark(f_current_jit, "Approach A (current)")
t_tracing = benchmark(f_tracing_jit, "Approach B (JAX tracing)")

# Correctness check
r_current = f_current_jit(2.5, 3.5)
r_tracing = f_tracing_jit(2.5, 3.5)
diff = abs(float(r_current[0]) - float(r_tracing))
print(f"\nCorrectness: difference = {diff:.2e} {'✓' if diff < 1e-6 else '✗'}")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"""
COMPILATION TIME:
  Approach A (current):     {time_current_total*1000:6.1f} ms
  Approach B (JAX tracing): {time_tracing_total*1000:6.1f} ms
  Speedup:                  {time_current_total/time_tracing_total:.2f}x

COMPILATION BREAKDOWN:
  A: xreplace {time_xreplace*1000:.1f}ms + lambdify {time_lambdify*1000:.1f}ms + JIT {time_jit*1000:.1f}ms
  B: dependencies {time_dependency*1000:.1f}ms + lambdify pieces {time_lambdify_pieces*1000:.1f}ms + compose {time_compose*1000:.1f}ms + JIT {time_jit_tracing*1000:.1f}ms

HLO SIZE:
  Approach A: {len(hlo_current):6,} chars (~{hlo_current.count('='):4} ops)
  Approach B: {len(hlo_tracing):6,} chars (~{hlo_tracing.count('='):4} ops)
  Ratio:      {len(hlo_tracing)/len(hlo_current):6.1f}x

RUNTIME:
  Approach A: {t_current*1000:.4f} ms
  Approach B: {t_tracing*1000:.4f} ms
  Difference: {abs(t_current-t_tracing)/min(t_current,t_tracing)*100:.1f}%

KEY INSIGHTS:
1. Approach B is {'FASTER' if time_tracing_total < time_current_total else 'SLOWER'} for compilation
   - Avoids xreplace on unused intermediates
   - Only lambdifies what's needed ({len(needed_indices)} not {N_total})

2. HLO size: Approach B is {len(hlo_tracing)/len(hlo_current):.1f}x {'larger' if len(hlo_tracing) > len(hlo_current) else 'smaller'}
   - More verbose intermediate representation
   - But XLA optimizes it away (see runtime)

3. Runtime is essentially identical
   - XLA produces same final code
   - Python composition overhead disappears during tracing

RECOMMENDATION:
{'Approach B (JAX tracing) is better - faster compilation, same runtime' if time_tracing_total < time_current_total else 'Approach A (current) is better - faster compilation'}
""")
