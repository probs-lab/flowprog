#!/usr/bin/env python3
"""
Test script to compare NumPy vs JAX backend performance for flowprog models.

This script creates a synthetic model with complex expressions including
Piecewise and Max functions (similar to the energy model) and benchmarks
both compilation backends.
"""

import time
import numpy as np
import sympy as sy
from flowprog.imperative_model import Model, Process, Object
from rdflib import URIRef

# Shorthand for object type
MASS = URIRef("http://qudt.org/vocab/quantitykind/Mass")
ENERGY = URIRef("http://qudt.org/vocab/quantitykind/Energy")


def create_test_model():
    """Create a test model similar to the energy model with complex expressions."""

    # Define processes (similar to energy model)
    processes = [
        Process("Producer1", produces=["intermediate"], consumes=["input1"]),
        Process("Producer2", produces=["intermediate"], consumes=["input2"]),
        Process("Consumer1", produces=["output"], consumes=["intermediate"]),
    ]

    objects = [
        Object("input1", MASS, has_market=False),
        Object("input2", MASS, has_market=False),
        Object("intermediate", ENERGY, has_market=True),
        Object("output", MASS, has_market=False),
    ]

    model = Model(processes, objects)

    # Create demand symbol
    demand = sy.Symbol("d")
    allocation_factor = sy.Symbol("a")
    supply_limit = sy.Symbol("S")

    # Pull production with allocation (creates complex expressions)
    result1 = model.pull_production(
        "output", demand, until_objects=["intermediate"],
        allocate_backwards={
            "intermediate": {
                "Producer1": allocation_factor,
                "Producer2": 1 - allocation_factor,
            }
        }
    )
    model.add(result1, label="Demand propagation with allocation")

    # Add supply with limit (creates Piecewise expressions)
    supply_result = model.pull_process_output("Producer1", "intermediate", supply_limit)
    limited_supply = model.limit(
        supply_result,
        expr=model.expr("ProcessOutput", process_id="Producer1", object_id="intermediate"),
        limit=model.expr("Consumption", object_id="intermediate"),
    )
    model.add(limited_supply, label="Limited supply")

    # Add deficit-based supply (creates Max expressions)
    model.add(
        model.pull_process_output(
            "Producer2",
            "intermediate",
            model.object_production_deficit("intermediate")
        ),
        label="Deficit-based supply"
    )

    return model


def benchmark_compilation(model, recipe_data, num_compilations=10):
    """Benchmark the compilation time for both backends."""

    print("=" * 70)
    print("COMPILATION BENCHMARK")
    print("=" * 70)

    # Test NumPy backend compilation
    print("\nNumPy backend compilation:")
    numpy_times = []
    for i in range(num_compilations):
        start = time.perf_counter()
        func_numpy = model.lambdify(data=recipe_data, backend='numpy')
        end = time.perf_counter()
        numpy_times.append(end - start)
        if i == 0:
            print(f"  First compilation: {numpy_times[0]*1000:.2f} ms")

    avg_numpy = np.mean(numpy_times[1:]) if len(numpy_times) > 1 else numpy_times[0]
    print(f"  Average (excluding first): {avg_numpy*1000:.2f} ms")

    # Test JAX backend compilation
    print("\nJAX backend compilation:")
    jax_times = []
    for i in range(num_compilations):
        start = time.perf_counter()
        func_jax = model.lambdify(data=recipe_data, backend='jax')
        end = time.perf_counter()
        jax_times.append(end - start)
        if i == 0:
            print(f"  First compilation: {jax_times[0]*1000:.2f} ms")

    avg_jax = np.mean(jax_times[1:]) if len(jax_times) > 1 else jax_times[0]
    print(f"  Average (excluding first): {avg_jax*1000:.2f} ms")

    return func_numpy, func_jax


def benchmark_evaluation(func_numpy, func_jax, test_params, num_iterations=1000):
    """Benchmark the evaluation time for both backends."""

    print("\n" + "=" * 70)
    print("EVALUATION BENCHMARK")
    print("=" * 70)

    # NumPy backend evaluation
    print(f"\nNumPy backend ({num_iterations} evaluations):")

    # Warm-up
    for _ in range(10):
        result_numpy = func_numpy(test_params)

    start = time.perf_counter()
    for _ in range(num_iterations):
        result_numpy = func_numpy(test_params)
    end = time.perf_counter()

    numpy_time = (end - start) / num_iterations
    print(f"  Average time per evaluation: {numpy_time*1000:.3f} ms")
    print(f"  Total time for {num_iterations} evaluations: {(end-start)*1000:.1f} ms")

    # JAX backend evaluation
    print(f"\nJAX backend ({num_iterations} evaluations):")

    # Warm-up (important for JIT)
    for _ in range(10):
        result_jax = func_jax(test_params)

    start = time.perf_counter()
    for _ in range(num_iterations):
        result_jax = func_jax(test_params)
    end = time.perf_counter()

    jax_time = (end - start) / num_iterations
    print(f"  Average time per evaluation: {jax_time*1000:.3f} ms")
    print(f"  Total time for {num_iterations} evaluations: {(end-start)*1000:.1f} ms")

    # Calculate speedup
    speedup = numpy_time / jax_time
    print(f"\n  Speedup: {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than NumPy")

    return result_numpy, result_jax, speedup


def verify_correctness(result_numpy, result_jax, tolerance=1e-6):
    """Verify that both backends produce the same results."""

    print("\n" + "=" * 70)
    print("CORRECTNESS VERIFICATION")
    print("=" * 70)

    # Compare a few sample values
    all_match = True
    mismatches = []

    for key in list(result_numpy.keys())[:5]:  # Check first 5 flows
        val_numpy = result_numpy[key]
        val_jax = result_jax[key]

        # Convert JAX array to float if needed
        if hasattr(val_jax, 'item'):
            val_jax = float(val_jax)

        diff = abs(val_numpy - val_jax)
        match = diff < tolerance

        if not match:
            all_match = False
            mismatches.append((key, val_numpy, val_jax, diff))

        print(f"\n  Flow {key[:16]}...:")
        print(f"    NumPy: {val_numpy:.6f}")
        print(f"    JAX:   {val_jax:.6f}")
        print(f"    Diff:  {diff:.2e} {'✓' if match else '✗'}")

    if all_match:
        print(f"\n✓ All tested values match within tolerance ({tolerance})")
    else:
        print(f"\n✗ Found {len(mismatches)} mismatches")

    return all_match


def main():
    """Run the benchmark."""

    print("\n" + "=" * 70)
    print("JAX Backend Proof of Concept for flowprog")
    print("=" * 70)

    # Create test model
    print("\nCreating test model...")
    model = create_test_model()

    # Define recipe data (process coefficients)
    recipe_data = {
        model.S[2, 0]: 1.5,  # Producer1 output coefficient
        model.U[0, 0]: 2.0,  # Producer1 input coefficient
        model.S[2, 1]: 1.2,  # Producer2 output coefficient
        model.U[1, 1]: 1.8,  # Producer2 input coefficient
        model.S[3, 2]: 1.0,  # Consumer1 output coefficient
        model.U[2, 2]: 3.0,  # Consumer1 input coefficient
    }

    print(f"  Processes: {len(model.processes)}")
    print(f"  Objects: {len(model.objects)}")
    print(f"  Intermediate expressions: {len(model._intermediates)}")

    # Show a sample expression to demonstrate complexity
    if model._intermediates:
        sample_sym, sample_expr, sample_desc = model._intermediates[0]
        print(f"\n  Sample expression ({sample_desc}):")
        print(f"    {sample_sym} = {sample_expr}")

    # Benchmark compilation
    func_numpy, func_jax = benchmark_compilation(model, recipe_data, num_compilations=5)

    # Test parameters
    test_params = {
        'd': 10.0,
        'a': 0.6,
        'S': 5.0,
    }

    print("\nTest parameters:")
    for k, v in test_params.items():
        print(f"  {k} = {v}")

    # Benchmark evaluation
    result_numpy, result_jax, speedup = benchmark_evaluation(
        func_numpy, func_jax, test_params, num_iterations=1000
    )

    # Verify correctness
    verify_correctness(result_numpy, result_jax)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nJAX backend is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than NumPy")
    print("\nKey benefits of JAX backend:")
    print("  ✓ Better handling of Piecewise expressions (no evaluate-all-branches)")
    print("  ✓ JIT compilation for optimized execution")
    print("  ✓ Easy installation (pip only, no compiler needed)")
    print("  ✓ Future-proof (GPU/TPU support available)")
    print("\nLimitations:")
    print("  - First evaluation includes JIT compilation overhead")
    print("  - Best suited for repeated evaluations (amortizes compilation cost)")
    print("=" * 70)


if __name__ == "__main__":
    main()
