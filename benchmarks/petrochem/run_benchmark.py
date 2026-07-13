#!/usr/bin/env python3
"""Build the petrochemicals model and check it reproduces the 21 reference
scenarios in model_data.json.

Usage:
    python benchmarks/petrochem/run_benchmark.py

This is a real-world model (143 processes, 111 objects, ~240 free
parameters) extracted from the C-THRU Global Petrochemicals Calculator
(https://github.com/ricklupton/global-petrochemicals-calculator), used here as
a complex realistic test case / benchmark for flowprog -- see README.md.
"""

import sys
import time

import numpy as np

from structure import load_data, build_structure
from model import define_model

RTOL = 1e-6
ATOL = 1e-3


def build():
    data = load_data()

    t0 = time.perf_counter()
    model_builder, recipe_data = build_structure(data)
    t1 = time.perf_counter()

    other_results = define_model(
        model_builder,
        recipe_data,
        data["processes_with_process_emissions"],
    )
    t2 = time.perf_counter()

    model = model_builder.build(recipe_data)
    t3 = time.perf_counter()

    # 'math' avoids sympy's numpy printer, which is pathologically slow (and
    # per flowprog's own docs, occasionally incorrect) printing the nested
    # Piecewise expressions this model produces. Scenarios are evaluated one
    # at a time, so there's no vectorisation to lose.
    func = model.lambdify(expressions=other_results, modules="math")
    t4 = time.perf_counter()

    print(f"structure : {t1 - t0:6.2f}s")
    print(f"build     : {t2 - t1:6.2f}s")
    print(f"compile   : {t3 - t2:6.2f}s")
    print(f"lambdify  : {t4 - t3:6.2f}s")

    return data, func


def get_model_output(func, params):
    # Some invalid division-by-zero warnings arise in branches of piecewise
    # expressions that aren't actually reached; safe to ignore here.
    with np.errstate(invalid="ignore"):
        result = func(params)
    return {
        k: float(v) if isinstance(v, np.ndarray) and v.ndim == 0 else v
        for k, v in result.items()
    }


def verify(data, func):
    n_checked = 0
    n_failed = 0
    t0 = time.perf_counter()
    for name, case in data["scenarios"].items():
        expected = case["results"]
        actual = get_model_output(func, case["params"])

        if set(expected) != set(actual):
            n_failed += 1
            print(f"[{name}] MISMATCHED KEYS")
            print(f"  missing from actual: {sorted(set(expected) - set(actual))}")
            print(f"  extra in actual:     {sorted(set(actual) - set(expected))}")
            continue

        bad = {}
        for k in expected:
            if not np.isclose(expected[k], actual[k], rtol=RTOL, atol=ATOL):
                bad[k] = (expected[k], actual[k])
        n_checked += 1
        if bad:
            n_failed += 1
            print(f"[{name}] {len(bad)} value(s) differ:")
            for k, (exp, act) in bad.items():
                print(f"  {k}: expected {exp!r}, got {act!r}")
    t1 = time.perf_counter()

    print(f"eval x{len(data['scenarios'])}: {t1 - t0:6.2f}s")
    print(f"\n{n_checked - n_failed}/{n_checked} scenarios matched")
    return n_failed == 0


def main():
    data, func = build()
    ok = verify(data, func)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
