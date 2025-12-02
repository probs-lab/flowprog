#!/usr/bin/env python3
"""
Inspect the actual lambdified function to see if intermediates are included.
"""

import sympy as sy
import numpy as np

print("=" * 70)
print("LAMBDIFY CSE INTERMEDIATE INSPECTION")
print("=" * 70)

x = sy.Symbol('x')
y = sy.Symbol('y')

# Create intermediates (like flowprog does)
temp1 = sy.Symbol('temp1')
temp2 = sy.Symbol('temp2')
temp3 = sy.Symbol('temp3')

# Define intermediate expressions (expensive)
intermediate_exprs = [
    (temp1, sum(x**i for i in range(20))),  # Expensive: 20 terms
    (temp2, sum(y**i for i in range(20))),  # Expensive: 20 terms
    (temp3, sum((x+y)**i for i in range(20))),  # Expensive: 20 terms
]

print("\nIntermediate expressions:")
for sym, expr in intermediate_exprs:
    print(f"  {sym} = <expensive: {len(str(expr))} chars>")

# Test 1: Output uses all intermediates
print("\n" + "=" * 70)
print("TEST 1: Output uses ALL intermediates")
print("=" * 70)

output1 = [temp1 + temp2 + temp3]
f1 = sy.lambdify([x, y], output1, cse=lambda e: (intermediate_exprs, e))

print("Inspecting compiled function:")
print(f"  Name: {f1.__code__.co_name}")
print(f"  # variables: {f1.__code__.co_nlocals}")
print(f"  Variables: {f1.__code__.co_varnames}")

# Look at the actual source (if available via dill/inspect)
import dis
print("\nDisassembly (first 30 instructions):")
dis.dis(f1, depth=1)

# Test 2: Output uses NO intermediates
print("\n" + "=" * 70)
print("TEST 2: Output uses NO intermediates")
print("=" * 70)

output2 = [x]  # Trivial, doesn't use temp1, temp2, or temp3
f2 = sy.lambdify([x, y], output2, cse=lambda e: (intermediate_exprs, e))

print("Inspecting compiled function:")
print(f"  Name: {f2.__code__.co_name}")
print(f"  # variables: {f2.__code__.co_nlocals}")
print(f"  Variables: {f2.__code__.co_varnames}")

print("\n⚠️  Even though output doesn't use intermediates:")
print(f"    Function still has {f2.__code__.co_nlocals} variables")
print(f"    (3 intermediates + inputs + internals)")

# Test 3: Performance comparison
print("\n" + "=" * 70)
print("TEST 3: Performance comparison")
print("=" * 70)

import time

# With intermediates (even though unused)
output_trivial = [x]
f_with_intermediates = sy.lambdify([x, y], output_trivial, cse=lambda e: (intermediate_exprs, e))

# Without intermediates
f_without_intermediates = sy.lambdify([x, y], output_trivial)

print("\nEvaluating trivial expression (just return x)...")

# Warm up
for _ in range(10):
    f_with_intermediates(2.5, 3.5)
    f_without_intermediates(2.5, 3.5)

# Benchmark with intermediates
start = time.perf_counter()
for _ in range(10000):
    result = f_with_intermediates(2.5, 3.5)
time_with = (time.perf_counter() - start) / 10000

# Benchmark without intermediates
start = time.perf_counter()
for _ in range(10000):
    result = f_without_intermediates(2.5, 3.5)
time_without = (time.perf_counter() - start) / 10000

print(f"\nWith unused intermediates:    {time_with*1000000:.2f} μs")
print(f"Without intermediates:        {time_without*1000000:.2f} μs")
print(f"Overhead from unused CSE:     {((time_with/time_without)-1)*100:.1f}%")

if time_with > time_without * 1.5:
    print("\n⚠️  SIGNIFICANT OVERHEAD from evaluating unused intermediates!")
else:
    print("\n✓ Overhead is minimal (Python optimizes away unused code)")

# Test 4: Check if SymPy's CSE actually evaluates them
print("\n" + "=" * 70)
print("TEST 4: Do intermediates actually get evaluated?")
print("=" * 70)

# Create an intermediate that would error if evaluated
error_intermediate = [
    (temp1, 1 / (x - 2.5)),  # Division by zero when x=2.5
]

output_safe = [y]  # Doesn't use temp1

try:
    f_risky = sy.lambdify([x, y], output_safe, cse=lambda e: (error_intermediate, e))
    result = f_risky(2.5, 10.0)  # This would error if temp1 is evaluated
    print("✓ Success! Intermediate was NOT evaluated")
    print(f"  Result: {result}")
except ZeroDivisionError:
    print("✗ Failed! Intermediate WAS evaluated (division by zero)")
    print("  Even though the output doesn't use it!")
