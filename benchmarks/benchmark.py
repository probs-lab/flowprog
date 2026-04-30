"""
Flowprog performance benchmark.

Usage:
    python benchmarks/benchmark.py              # both cases, all sizes
    python benchmarks/benchmark.py --case a     # plain pull_production only
    python benchmarks/benchmark.py --case b     # limit() steps only
    python benchmarks/benchmark.py --quick      # small sizes for smoke-testing
    python benchmarks/benchmark.py --timeout 30 # per-phase timeout in seconds

Case (a): Plain models using pull_production only
──────────────────────────────────────────────────
Two model shapes:
  chain  : linear chain P0→O0←P1→O1←…←P(N-1)→O(N-1)
  fan    : one root process consuming N leaf processes in parallel

Flowprog phases timed separately:
  build    : ModelBuilder() + pull_production() + add()
  compile  : builder.build(recipe)  →  SympyModel
  lambdify : model.lambdify()        →  numpy or math function
  eval     : calling the function with demand=1.0

Comparison (scipy):  matrix_build | matrix_solve
Comparison (brightway, if installed):  bw_db_setup | bw_lci_solve

Case (b): Chain-of-processes limit model with K limit steps
─────────────────────────────────────────────────────────────
K+1 processes form a linear supply chain.  Each of the K limit steps
uses object_production_deficit() to drive the next process, adding a
Piecewise layer that references accumulated state from all prior steps.
This causes expression complexity to grow rapidly with K.

Capacity limits are symbolic parameters (cap_1, cap_2, …), not numeric.

Structure measurements are included for each K to quantify expression growth:
  raw_ops_total    : total SymPy operation count across all compiled _values slots
  raw_ops_max_slot : maximum op count in any single slot
  n_intermediates  : number of named intermediate symbols created
  n_free_symbols   : number of free (non-Indexed) parameters in compiled model

Timeouts (--timeout flag, default 30s) are applied per-phase using
concurrent.futures threads.  A timed-out phase is recorded as NaN.
Once an approach times out it is skipped for all larger sizes/steps.
"""

import csv
import concurrent.futures
import os
import timeit
import argparse
from datetime import datetime
from typing import Optional

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import sympy as sy
from rdflib import URIRef

from flowprog.model_builder import ModelBuilder
from flowprog.model_structure import Process, Object

MASS = URIRef("http://qudt.org/vocab/quantitykind/Mass")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# ─── timing helpers ───────────────────────────────────────────────────────────

def _tmin(fn, *, reps: int = 3, calls: int = 1) -> float:
    """Return minimum wall-clock time per call (seconds)."""
    return min(timeit.repeat(fn, number=calls, repeat=reps)) / calls


def _with_timeout(fn, timeout_seconds: float):
    """Run fn() in a background thread; return result or None on timeout/error.

    The background thread is not cancelled on timeout — it continues running
    until completion.  This is acceptable for benchmarking purposes.
    """
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(fn)
    try:
        return future.result(timeout=timeout_seconds)
    except concurrent.futures.TimeoutError:
        return None
    except Exception:
        return None
    finally:
        executor.shutdown(wait=False)


# ─── structure measurement ────────────────────────────────────────────────────

def measure_structure(model) -> dict:
    """Count expression complexity in a compiled SympyModel.

    Operates on model._values and model._intermediates without lambdifying.

    Returns:
      raw_ops_total    : sum of sy.count_ops across all non-zero _values slots
      raw_ops_max_slot : maximum op count in any single slot
      n_intermediates  : number of named intermediate symbols
      n_free_symbols   : number of free non-Indexed parameters
    """
    raw_ops_total = 0
    raw_ops_max_slot = 0
    for _sym, val in model._values.items():
        if val == sy.S.Zero:
            continue
        n = sy.count_ops(val)
        raw_ops_total += n
        raw_ops_max_slot = max(raw_ops_max_slot, n)

    n_intermediates = len(model._intermediates)

    all_free: set = set()
    for val in model._values.values():
        if val != sy.S.Zero:
            all_free |= val.free_symbols
    for _sym, expr, _ in model._intermediates:
        all_free |= expr.free_symbols
    n_free_symbols = len({s for s in all_free if not isinstance(s, sy.Indexed)})

    return dict(
        raw_ops_total=raw_ops_total,
        raw_ops_max_slot=raw_ops_max_slot,
        n_intermediates=n_intermediates,
        n_free_symbols=n_free_symbols,
    )


# ─── CSV output ───────────────────────────────────────────────────────────────

def _save_csv(rows: list[dict], filename: str) -> str:
    """Write rows to benchmarks/results/<filename>.  Returns path."""
    if not rows:
        return ""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys(), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            # Replace None with empty string for CSV clarity
            writer.writerow({k: ("" if v is None else v) for k, v in row.items()})
    return path


# ══════════════════════════════════════════════════════════════════════════════
# Case (a): plain pull_production — chain and fan variants
# ══════════════════════════════════════════════════════════════════════════════

def _make_chain(n: int):
    """Linear chain of N processes + symbol-keyed recipe dict."""
    objects = [Object(f"O{i}", MASS, has_market=True) for i in range(n)]
    processes = [
        Process(f"P{i}", produces=[f"O{i}"], consumes=[f"O{i-1}"] if i > 0 else [])
        for i in range(n)
    ]
    tmp = ModelBuilder(processes, objects)
    recipe = {tmp.S[i, i]: 1.0 for i in range(n)}
    recipe.update({tmp.U[i - 1, i]: 0.8 for i in range(1, n)})
    return processes, objects, recipe


def _make_fan(n_leaves: int):
    """Fan: one root process consuming N leaf processes in parallel.

    Structure:
      P_root   : produces "output", consumes "leaf_0".."leaf_{N-1}"
      P_leaf_i : produces "leaf_i"
    """
    objects = [Object("output", MASS, has_market=True)] + [
        Object(f"leaf_{i}", MASS, has_market=True) for i in range(n_leaves)
    ]
    processes = [
        Process("P_root", produces=["output"],
                consumes=[f"leaf_{i}" for i in range(n_leaves)])
    ] + [
        Process(f"P_leaf_{i}", produces=[f"leaf_{i}"], consumes=[])
        for i in range(n_leaves)
    ]
    tmp = ModelBuilder(processes, objects)
    recipe = {tmp.S[0, 0]: 1.0}
    recipe.update({tmp.U[i + 1, 0]: 0.8 for i in range(n_leaves)})
    recipe.update({tmp.S[i + 1, i + 1]: 1.0 for i in range(n_leaves)})
    return processes, objects, recipe


DEMAND_A = sy.Symbol("demand", positive=True)


def _bench_plain_flowprog(processes, objects, recipe, final_object: str,
                           timeout: float) -> dict:
    """Time flowprog phases for any model; returns partial dict on timeout."""
    result: dict = {}

    def do_build():
        b = ModelBuilder(processes, objects)
        b.add(b.pull_production(final_object, DEMAND_A))
        return b

    result["build"] = _tmin(do_build, reps=5)
    builder = do_build()

    result["compile"] = _tmin(lambda: builder.build(recipe), reps=5)
    model = builder.build(recipe)
    result.update(measure_structure(model))

    # lambdify phases — each wrapped with a timeout
    def do_lambdify_np():
        t = _tmin(lambda: model.lambdify(), reps=3)
        fn = model.lambdify()
        return t, fn

    res = _with_timeout(do_lambdify_np, timeout)
    if res is None:
        result["lambdify_np"] = None
        result["eval_np"] = None
    else:
        result["lambdify_np"], fn_np = res
        result["eval_np"] = _tmin(lambda: fn_np({DEMAND_A: 1.0}), reps=5, calls=50)

    def do_lambdify_math():
        t = _tmin(lambda: model.lambdify(modules="math"), reps=3)
        fn = model.lambdify(modules="math")
        return t, fn

    res = _with_timeout(do_lambdify_math, timeout)
    if res is None:
        result["lambdify_math"] = None
        result["eval_math"] = None
    else:
        result["lambdify_math"], fn_math = res
        result["eval_math"] = _tmin(
            lambda: fn_math({DEMAND_A: 1.0}), reps=5, calls=50
        )

    return result


def bench_plain_flowprog_chain(n: int, timeout: float) -> Optional[dict]:
    processes, objects, recipe = _make_chain(n)
    r = _bench_plain_flowprog(processes, objects, recipe, f"O{n - 1}", timeout)
    return dict(n=n, **r)


def bench_plain_flowprog_fan(n_leaves: int, timeout: float) -> Optional[dict]:
    processes, objects, recipe = _make_fan(n_leaves)
    r = _bench_plain_flowprog(processes, objects, recipe, "output", timeout)
    return dict(n=n_leaves, **r)


# ── Scipy ─────────────────────────────────────────────────────────────────────

def bench_plain_scipy(n: int) -> dict:
    def do_build():
        rows, cols, vals = [], [], []
        for i in range(n):
            rows.append(i); cols.append(i); vals.append(1.0)
            if i > 0:
                rows.append(i - 1); cols.append(i); vals.append(-0.8)
        return scipy.sparse.csc_matrix((vals, (rows, cols)), shape=(n, n))

    t_build = _tmin(do_build, reps=5)
    A = do_build()
    f = np.zeros(n)
    f[n - 1] = 1.0
    t_solve = _tmin(lambda: scipy.sparse.linalg.spsolve(A, f), reps=5, calls=10)
    return dict(n=n, matrix_build=t_build, matrix_solve=t_solve)


# ── Brightway (optional) ──────────────────────────────────────────────────────

def _try_import_bw():
    try:
        import bw2data
        import bw2calc
        return bw2data, bw2calc
    except ImportError:
        return None, None


def bench_plain_bw25(n: int, project_name: str = "flowprog_benchmark") -> Optional[dict]:
    bw2data, bw2calc = _try_import_bw()
    if bw2data is None:
        return None

    db_name = f"chain_{n}"

    def do_db_setup():
        bw2data.projects.set_current(project_name)
        if db_name in bw2data.databases:
            del bw2data.databases[db_name]
        db = bw2data.Database(db_name)
        data = {}
        for i in range(n):
            key = (db_name, f"P{i}")
            exchanges = [{"input": key, "output": key, "amount": 1.0, "type": "production"}]
            if i > 0:
                exchanges.append({
                    "input": (db_name, f"P{i-1}"),
                    "output": key,
                    "amount": 0.8,
                    "type": "technosphere",
                })
            data[key] = {"name": f"P{i}", "unit": "unit", "exchanges": exchanges}
        db.write(data)
        return db

    t_db_setup = _tmin(do_db_setup, reps=3)
    db = do_db_setup()
    final_act = db.get(f"P{n - 1}")

    def do_lci():
        lca = bw2calc.LCA({final_act: 1.0})
        lca.lci()
        return lca

    t_lci = _tmin(do_lci, reps=3)
    return dict(n=n, bw_db_setup=t_db_setup, bw_lci_solve=t_lci)


# ── Run case (a) ──────────────────────────────────────────────────────────────

def run_case_a(sizes, verbose=True, timeout=30.0):
    bw2data, _ = _try_import_bw()
    has_bw = bw2data is not None

    chain_rows, fan_rows, sp_rows, bw_rows = [], [], [], []
    chain_alive = fan_alive = scipy_alive = bw_alive = True

    def _fp_row_str(r):
        np_str = (f"{r['lambdify_np']*1e3:6.1f}ms" if r.get("lambdify_np") is not None
                  else " timeout")
        math_str = (f"{r['lambdify_math']*1e3:6.1f}ms" if r.get("lambdify_math") is not None
                    else " timeout")
        enp = (f"{r['eval_np']*1e6:6.1f}µs" if r.get("eval_np") is not None else "   ---  ")
        em = (f"{r['eval_math']*1e6:6.1f}µs" if r.get("eval_math") is not None else "   ---  ")
        return (
            f"build={r['build']*1e3:5.1f}ms  compile={r['compile']*1e3:5.1f}ms  "
            f"lambdify np={np_str} math={math_str}  "
            f"eval np={enp} math={em}  "
            f"ops={r['raw_ops_total']:5d} intermediates={r['n_intermediates']:3d}"
        )

    for n in sizes:
        # --- chain ---
        if chain_alive:
            if verbose:
                print(f"  N={n:4d} chain...", end=" ", flush=True)
            r = _with_timeout(lambda n=n: bench_plain_flowprog_chain(n, timeout), timeout * 4)
            if r is None:
                chain_alive = False
                if verbose:
                    print("TIMEOUT — skipping chain for larger N")
            else:
                chain_rows.append(r)
                if verbose:
                    print(_fp_row_str(r))

        # --- fan ---
        if fan_alive:
            if verbose:
                print(f"  N={n:4d} fan  ...", end=" ", flush=True)
            r = _with_timeout(lambda n=n: bench_plain_flowprog_fan(n, timeout), timeout * 4)
            if r is None:
                fan_alive = False
                if verbose:
                    print("TIMEOUT — skipping fan for larger N")
            else:
                fan_rows.append(r)
                if verbose:
                    print(_fp_row_str(r))

        # --- scipy ---
        if scipy_alive:
            try:
                sp = bench_plain_scipy(n)
                sp_rows.append(sp)
                if verbose:
                    print(
                        f"  N={n:4d} scipy   build={sp['matrix_build']*1e6:5.1f}µs  "
                        f"solve={sp['matrix_solve']*1e6:5.1f}µs"
                    )
            except Exception:
                scipy_alive = False

        # --- brightway ---
        if has_bw and bw_alive:
            try:
                bw = bench_plain_bw25(n)
                if bw:
                    bw_rows.append(bw)
                    if verbose:
                        print(
                            f"  N={n:4d} bw      db_setup={bw['bw_db_setup']*1e3:6.1f}ms  "
                            f"lci={bw['bw_lci_solve']*1e3:6.1f}ms"
                        )
            except Exception as e:
                bw_alive = False
                if verbose:
                    print(f"  brightway error: {e}")

        if verbose:
            print()

    return chain_rows, fan_rows, sp_rows, bw_rows


# ══════════════════════════════════════════════════════════════════════════════
# Case (b): chain-of-processes limit model
# ══════════════════════════════════════════════════════════════════════════════

def _make_chain_limit_model(k: int):
    """Chain-of-processes model with K limit steps.

    Capacity limits are symbolic: cap_1, cap_2, …, cap_K.

    Structure (K=2 example):
      Objects  : output, intermediate_1, intermediate_2
      Processes: P1 (output ← intermediate_1)
                 P2 (intermediate_1 ← intermediate_2)
                 Pfinal (intermediate_2 ← [nothing])

    Steps:
      1      : limit(pull_production("output", demand,
                       until_objects=["intermediate_1"]),
                     expr("ProcessOutput", P1, "output"), cap_1)
      2..K   : limit(pull_production("intermediate_{i-1}",
                       object_production_deficit("intermediate_{i-1}"),
                       until_objects=["intermediate_i"]),
                     expr("ProcessOutput", P_i, "intermediate_{i-1}"), cap_i)
      K+1    : pull_production("intermediate_K",
                               object_production_deficit("intermediate_K"))
    """
    object_names = ["output"] + [f"intermediate_{i}" for i in range(1, k + 1)]
    objects = [Object(name, MASS, has_market=True) for name in object_names]

    processes = [
        Process(f"P{i + 1}", produces=[object_names[i]], consumes=[object_names[i + 1]])
        for i in range(k)
    ] + [
        Process("Pfinal", produces=[object_names[k]], consumes=[])
    ]

    b = ModelBuilder(processes, objects)

    demand_sym = sy.Symbol("demand", positive=True)

    # Step 1: limit on P1's output by symbolic cap_1
    cap1 = sy.Symbol("cap_1", positive=True)
    act1 = b.pull_production("output", demand_sym, until_objects=["intermediate_1"])
    b.add(b.limit(act1, b.expr("ProcessOutput", process_id="P1", object_id="output"), cap1))

    # Steps 2..K
    for i in range(2, k + 1):
        obj_name = object_names[i - 1]
        next_obj = object_names[i]
        pid = f"P{i}"
        cap_i = sy.Symbol(f"cap_{i}", positive=True)
        deficit = b.object_production_deficit(obj_name)
        act = b.pull_production(obj_name, deficit, until_objects=[next_obj])
        b.add(b.limit(act, b.expr("ProcessOutput", process_id=pid, object_id=obj_name), cap_i))

    # Final unconstrained step
    b.add(b.pull_production(object_names[k], b.object_production_deficit(object_names[k])))

    # Recipe: process i (0-based) produces object i, consumes object i+1
    recipe = {b.S[i, i]: 1.0 for i in range(k + 1)}
    recipe.update({b.U[i + 1, i]: 0.8 for i in range(k)})

    return b, recipe


def _chain_limit_params(k: int) -> dict:
    """Sample parameter dict for evaluating a K-step chain limit model."""
    params = {sy.Symbol("demand", positive=True): 1.0}
    params.update({sy.Symbol(f"cap_{i}", positive=True): float(10 + i) for i in range(1, k + 1)})
    return params


def run_case_b(step_counts, verbose=True, timeout=30.0):
    """Run case (b) for each K in step_counts.

    Tracks per-phase bail: lambdify_np and lambdify_math are stopped independently
    once they time out.
    """
    rows = []
    np_alive = True
    math_alive = True

    for k in step_counts:
        if verbose:
            print(f"  K={k:3d}...", end=" ", flush=True)

        # Build + compile + structure (fast, no timeout needed)
        t_build = _tmin(lambda k=k: _make_chain_limit_model(k), reps=3)
        builder, recipe = _make_chain_limit_model(k)
        t_compile = _tmin(lambda: builder.build(recipe), reps=3)
        model = builder.build(recipe)
        struct = measure_structure(model)

        # lambdify_np
        t_lambdify_np = None
        if np_alive:
            res = _with_timeout(
                lambda: _tmin(lambda: model.lambdify(), reps=1),
                timeout,
            )
            if res is None:
                np_alive = False
                t_lambdify_np = None
            else:
                t_lambdify_np = res

        # lambdify_math + eval_math (independent of np_alive)
        t_lambdify_math = None
        t_eval_math = None
        if math_alive:
            def _do_math_lambdify(model=model):
                t = _tmin(lambda: model.lambdify(modules="math"), reps=1)
                fn = model.lambdify(modules="math")
                return t, fn

            res = _with_timeout(_do_math_lambdify, timeout)
            if res is None:
                math_alive = False
            else:
                t_lambdify_math, fn_math = res
                params = _chain_limit_params(k)
                t_eval_math = _tmin(lambda: fn_math(params), reps=5, calls=50)

        row = dict(
            k=k,
            build=t_build,
            compile=t_compile,
            **struct,
            lambdify_np=t_lambdify_np,
            lambdify_math=t_lambdify_math,
            eval_math=t_eval_math,
        )
        rows.append(row)

        if verbose:
            def _fmt(v, unit, scale):
                return f"{v * scale:.2f}{unit}" if v is not None else "timeout"

            np_str = _fmt(t_lambdify_np, "ms", 1e3)
            math_str = _fmt(t_lambdify_math, "ms", 1e3)
            eval_str = _fmt(t_eval_math, "µs", 1e6)
            print(
                f"build={t_build*1e3:.2f}ms  compile={t_compile*1e3:.2f}ms  "
                f"ops={struct['raw_ops_total']:6d}  ints={struct['n_intermediates']:3d}  "
                f"free_syms={struct['n_free_symbols']:3d}  "
                f"lambdify np={np_str:>12s} math={math_str:>12s}  "
                f"eval(math)={eval_str}"
            )

        if not np_alive and not math_alive:
            if verbose:
                print("  (both lambdify approaches timed out — stopping)")
            break

    return rows


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

SIZES_FULL = [5, 10, 20, 50, 100, 200]
SIZES_QUICK = [5, 10, 20]
STEPS_FULL = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16]
STEPS_QUICK = [1, 2, 3, 4, 5]


def main():
    parser = argparse.ArgumentParser(description="Flowprog performance benchmark")
    parser.add_argument("--case", choices=["a", "b"], help="Run only this case")
    parser.add_argument("--quick", action="store_true", help="Smaller sizes for quick smoke-test")
    parser.add_argument(
        "--timeout", type=float, default=30.0,
        help="Per-phase timeout in seconds (default 30); timed-out phases recorded as blank",
    )
    args = parser.parse_args()

    sizes = SIZES_QUICK if args.quick else SIZES_FULL
    steps = STEPS_QUICK if args.quick else STEPS_FULL
    run_a = args.case in (None, "a")
    run_b = args.case in (None, "b")

    bw2data, _ = _try_import_bw()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 88)
    print("Flowprog performance benchmark")
    print(f"  timeout={args.timeout}s  quick={args.quick}  {ts}")
    print("=" * 88)

    if run_a:
        print()
        print("Case (a): Plain pull_production — chain and fan variants")
        print("-" * 88)
        print("Columns: build | compile | lambdify(np/math) | eval(np/math) | ops | intermediates")
        if bw2data:
            print("Brightway comparison included.")
        else:
            print("(brightway not installed — scipy matrix solver used as comparison)")
        print()

        chain_rows, fan_rows, sp_rows, bw_rows = run_case_a(
            sizes, verbose=True, timeout=args.timeout
        )

        # Save CSVs
        for rows, name in [
            (chain_rows, f"case_a_chain_{ts}.csv"),
            (fan_rows,   f"case_a_fan_{ts}.csv"),
            (sp_rows,    f"case_a_scipy_{ts}.csv"),
            (bw_rows,    f"case_a_bw_{ts}.csv"),
        ]:
            if rows:
                p = _save_csv(rows, name)
                print(f"  Saved {p}")

    if run_b:
        print()
        print("Case (b): Chain-of-processes limit model — K limit steps")
        print("-" * 88)
        print("Chain: P1(output←int1) → P2(int1←int2) → … → Pfinal(intK)")
        print("Capacity limits are symbolic parameters (cap_1, cap_2, …).")
        print("Columns: build | compile | ops | ints | free_syms | lambdify(np/math) | eval(math)")
        print(f"(per-phase timeout {args.timeout}s; timed-out phases recorded as blank)")
        print()

        b_rows = run_case_b(steps, verbose=True, timeout=args.timeout)

        if b_rows:
            p = _save_csv(b_rows, f"case_b_{ts}.csv")
            print(f"\n  Saved {p}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
