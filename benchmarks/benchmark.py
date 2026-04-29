"""
Flowprog performance benchmark.

Usage:
    python benchmarks/benchmark.py              # both cases, all sizes
    python benchmarks/benchmark.py --case a     # plain pull_production only
    python benchmarks/benchmark.py --case b     # limit() steps only
    python benchmarks/benchmark.py --quick      # small sizes for smoke-testing

Case (a): Plain models using pull_production only
──────────────────────────────────────────────────
Two model shapes:
  chain  : linear chain P0→O0←P1→O1←…←P(N-1)→O(N-1)
  fan    : one root process consuming N leaf processes in parallel

Each process Pi produces Oi and (in the chain) consumes O(i-1); S=1, U=0.8.
We demand 1 unit of the final object and trace backwards.

Flowprog phases timed separately:
  build    : ModelBuilder() + pull_production() + add()
  compile  : builder.build(recipe)  →  SympyModel
  lambdify : model.lambdify()        →  numpy function
  eval     : calling the numpy function with demand=1.0

Comparison (scipy, always available):
  matrix_build : build sparse technosphere matrix
  matrix_solve : scipy.sparse.linalg.spsolve

Comparison (brightway, if installed):
  bw_db_setup  : write activities to bw2 database
  bw_lci_solve : LCA() + lca.lci()

Case (b): Chain-of-processes limit model with K limit steps
─────────────────────────────────────────────────────────────
A realistic chain where:
  - K+1 processes form a linear supply chain
  - P_i produces "intermediate_{i-1}" (or "output" for i=1), consuming "intermediate_i"
  - Pfinal produces the last intermediate with no upstream inputs

Steps:
  step 1   : pull from "output" up to capacity of P1 (stopping at intermediate_1)
  step 2..K: satisfy remaining deficit of each intermediate up to P_{i}'s capacity
  step K+1 : satisfy remaining deficit of last intermediate (unconstrained)

Each limit() step adds a Piecewise layer referencing accumulated state from prior
steps (via object_production_deficit), so expression complexity grows with K.

Same four flowprog phases timed; no brightway comparison.
"""

import timeit
import argparse
from typing import Optional

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import sympy as sy
from rdflib import URIRef

from flowprog.model_builder import ModelBuilder
from flowprog.model_structure import Process, Object

MASS = URIRef("http://qudt.org/vocab/quantitykind/Mass")

# ─── timing helper ────────────────────────────────────────────────────────────

def _tmin(fn, *, reps: int = 3, calls: int = 1) -> float:
    """Return minimum wall-clock time per call (seconds)."""
    return min(timeit.repeat(fn, number=calls, repeat=reps)) / calls


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
    """Fan/tree: one root process consuming N leaf processes in parallel.

    Structure:
      P_root : produces "output", consumes "leaf_0".."leaf_{N-1}"
      P_leaf_i : produces "leaf_i"

    The pull_production("output", demand) fans out to all N leaf processes
    simultaneously, unlike the linear chain where it recurses depth-first.
    """
    objects = [Object("output", MASS, has_market=True)] + [
        Object(f"leaf_{i}", MASS, has_market=True) for i in range(n_leaves)
    ]
    processes = [
        Process("P_root", produces=["output"], consumes=[f"leaf_{i}" for i in range(n_leaves)])
    ] + [
        Process(f"P_leaf_{i}", produces=[f"leaf_{i}"], consumes=[]) for i in range(n_leaves)
    ]
    tmp = ModelBuilder(processes, objects)
    # P_root (proc index 0) produces "output" (obj index 0)
    recipe = {tmp.S[0, 0]: 1.0}
    # P_root consumes leaf_i (obj index i+1): U[i+1, 0] = 0.8
    recipe.update({tmp.U[i + 1, 0]: 0.8 for i in range(n_leaves)})
    # P_leaf_i (proc index i+1) produces leaf_i (obj index i+1): S[i+1, i+1] = 1.0
    recipe.update({tmp.S[i + 1, i + 1]: 1.0 for i in range(n_leaves)})
    return processes, objects, recipe


DEMAND_A = sy.Symbol("demand", positive=True)


def _bench_plain_flowprog(processes, objects, recipe, final_object: str) -> dict:
    """Time flowprog phases for any model specified by processes/objects/recipe."""

    def do_build():
        b = ModelBuilder(processes, objects)
        b.add(b.pull_production(final_object, DEMAND_A))
        return b

    t_build = _tmin(do_build, reps=5)
    builder = do_build()

    t_compile = _tmin(lambda: builder.build(recipe), reps=5)
    model = builder.build(recipe)

    t_lambdify_np = _tmin(lambda: model.lambdify(), reps=3)
    fn_np = model.lambdify()

    t_lambdify_math = _tmin(lambda: model.lambdify(modules="math"), reps=3)
    fn_math = model.lambdify(modules="math")

    t_eval_np = _tmin(lambda: fn_np({DEMAND_A: 1.0}), reps=5, calls=50)
    t_eval_math = _tmin(lambda: fn_math({DEMAND_A: 1.0}), reps=5, calls=50)

    return dict(
        build=t_build,
        compile=t_compile,
        lambdify_np=t_lambdify_np,
        lambdify_math=t_lambdify_math,
        eval_np=t_eval_np,
        eval_math=t_eval_math,
    )


def bench_plain_flowprog_chain(n: int) -> dict:
    """Time the flowprog phases for a plain N-process chain."""
    processes, objects, recipe = _make_chain(n)
    result = _bench_plain_flowprog(processes, objects, recipe, f"O{n - 1}")
    return dict(n=n, **result)


def bench_plain_flowprog_fan(n_leaves: int) -> dict:
    """Time the flowprog phases for a fan model with N leaf processes."""
    processes, objects, recipe = _make_fan(n_leaves)
    result = _bench_plain_flowprog(processes, objects, recipe, "output")
    return dict(n=n_leaves, **result)


# ── Scipy matrix comparison ───────────────────────────────────────────────────

def bench_plain_scipy(n: int) -> dict:
    """Time a scipy sparse matrix solve for an N-process chain.

    Technosphere matrix A:
      A[i,i] = 1.0   (production)
      A[i-1,i] = -0.8 (consumption of prior output)
    """
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


# ── Brightway comparison (optional) ──────────────────────────────────────────

def _try_import_bw():
    try:
        import bw2data
        import bw2calc
        return bw2data, bw2calc
    except ImportError:
        return None, None


def bench_plain_bw25(n: int, project_name: str = "flowprog_benchmark") -> Optional[dict]:
    """Time brightway25 for an equivalent N-process chain.

    Returns None if bw2data/bw2calc are not installed.
    """
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

    def do_lci_solve():
        lca = bw2calc.LCA({final_act: 1.0})
        lca.lci()
        return lca

    t_lci_solve = _tmin(do_lci_solve, reps=3)

    return dict(n=n, bw_db_setup=t_db_setup, bw_lci_solve=t_lci_solve)


# ── Run case (a) ─────────────────────────────────────────────────────────────

def run_case_a(sizes, verbose=True):
    bw2data, bw2calc = _try_import_bw()
    has_bw = bw2data is not None

    chain_rows = []
    fan_rows = []
    sp_rows = []
    bw_rows = []

    for n in sizes:
        if verbose:
            print(f"  N={n:4d} chain...", end=" ", flush=True)

        fp = bench_plain_flowprog_chain(n)
        sp = bench_plain_scipy(n)
        bw = bench_plain_bw25(n) if has_bw else None

        chain_rows.append(fp)
        sp_rows.append(sp)
        if bw:
            bw_rows.append(bw)

        if verbose:
            bw_str = (
                f"  | bw db_setup={bw['bw_db_setup']*1e3:.1f}ms lci={bw['bw_lci_solve']*1e3:.1f}ms"
                if bw else ""
            )
            print(
                f"build={fp['build']*1e3:6.1f}ms  compile={fp['compile']*1e3:6.1f}ms  "
                f"lambdify np={fp['lambdify_np']*1e3:6.1f}ms math={fp['lambdify_math']*1e3:6.1f}ms  "
                f"eval np={fp['eval_np']*1e6:6.1f}µs math={fp['eval_math']*1e6:6.1f}µs  "
                f"| scipy build={sp['matrix_build']*1e6:5.1f}µs solve={sp['matrix_solve']*1e6:5.1f}µs"
                + bw_str
            )

    if verbose:
        print()
        print("  Fan variant (N = number of leaf processes):")
    for n in sizes:
        if verbose:
            print(f"  N={n:4d} fan  ...", end=" ", flush=True)

        fp = bench_plain_flowprog_fan(n)
        fan_rows.append(fp)

        if verbose:
            print(
                f"build={fp['build']*1e3:6.1f}ms  compile={fp['compile']*1e3:6.1f}ms  "
                f"lambdify np={fp['lambdify_np']*1e3:6.1f}ms math={fp['lambdify_math']*1e3:6.1f}ms  "
                f"eval np={fp['eval_np']*1e6:6.1f}µs math={fp['eval_math']*1e6:6.1f}µs"
            )

    return chain_rows, fan_rows, sp_rows, bw_rows


# ══════════════════════════════════════════════════════════════════════════════
# Case (b): chain-of-processes limit model
# ══════════════════════════════════════════════════════════════════════════════

def _make_chain_limit_model(k: int):
    """Chain-of-processes model with K limit steps.

    Structure (K=2 example):
      Objects  : "output", "intermediate_1", "intermediate_2"
      Processes: P1 (output ← intermediate_1)
                 P2 (intermediate_1 ← intermediate_2)
                 Pfinal (intermediate_2 ← [nothing])

    Steps:
      1      : limit(pull_production("output", demand, until_objects=["intermediate_1"]),
                     expr("ProcessOutput", process_id="P1", object_id="output"), cap_1)
      2..K   : limit(pull_production("intermediate_{i-1}",
                                     object_production_deficit("intermediate_{i-1}"),
                                     until_objects=["intermediate_i"]),
                     expr("ProcessOutput", process_id="P_{i}", object_id="intermediate_{i-1}"),
                     cap_i)
      K+1    : pull_production("intermediate_K",
                               object_production_deficit("intermediate_K"))

    Each limit step references the accumulated production deficit of the
    previous intermediate, so expression complexity grows with K.
    """
    object_names = ["output"] + [f"intermediate_{i}" for i in range(1, k + 1)]
    objects = [Object(name, MASS, has_market=True) for name in object_names]

    # K limited processes in the chain + 1 unconstrained final process
    # Process i (0-indexed) = P_{i+1}: produces object_names[i], consumes object_names[i+1]
    # Process K = Pfinal: produces object_names[K], no consumption
    processes = [
        Process(f"P{i + 1}", produces=[object_names[i]], consumes=[object_names[i + 1]])
        for i in range(k)
    ] + [
        Process("Pfinal", produces=[object_names[k]], consumes=[])
    ]

    b = ModelBuilder(processes, objects)

    demand_sym = sy.Symbol("demand", positive=True)

    # Step 1: pull "output" up to capacity of P1
    act1 = b.pull_production("output", demand_sym, until_objects=["intermediate_1"])
    b.add(b.limit(act1, b.expr("ProcessOutput", process_id="P1", object_id="output"), 10.0))

    # Steps 2..K: satisfy remaining deficit at each intermediate
    for i in range(2, k + 1):
        obj_name = object_names[i - 1]       # "intermediate_{i-1}"
        next_obj = object_names[i]            # "intermediate_i"
        pid = f"P{i}"
        deficit = b.object_production_deficit(obj_name)
        act = b.pull_production(obj_name, deficit, until_objects=[next_obj])
        cap = 10.0 + i
        b.add(b.limit(act, b.expr("ProcessOutput", process_id=pid, object_id=obj_name), cap))

    # Final step: satisfy remaining deficit of last intermediate (unconstrained)
    b.add(b.pull_production(object_names[k], b.object_production_deficit(object_names[k])))

    # Recipe: process i (0-based) produces object i, consumes object i+1
    # S[i, i] = 1.0 for i in 0..K
    # U[i+1, i] = 0.8 for i in 0..K-1 (process i consumes next object)
    recipe = {b.S[i, i]: 1.0 for i in range(k + 1)}
    recipe.update({b.U[i + 1, i]: 0.8 for i in range(k)})

    return b, recipe


def bench_chain_limit_flowprog(k: int) -> dict:
    """Time flowprog phases for the chain-of-processes limit model with K limit steps."""

    t_build = _tmin(lambda: _make_chain_limit_model(k), reps=5)
    builder, recipe = _make_chain_limit_model(k)

    t_compile = _tmin(lambda: builder.build(recipe), reps=3)
    model = builder.build(recipe)

    t_lambdify_np = _tmin(lambda: model.lambdify(), reps=3)

    t_lambdify_math = _tmin(lambda: model.lambdify(modules="math"), reps=3)
    fn_math = model.lambdify(modules="math")

    demand_sym = sy.Symbol("demand", positive=True)
    params = {demand_sym: 1.0}
    t_eval_math = _tmin(lambda: fn_math(params), reps=5, calls=50)

    return dict(
        k=k,
        build=t_build,
        compile=t_compile,
        lambdify_np=t_lambdify_np,
        lambdify_math=t_lambdify_math,
        eval_math=t_eval_math,
    )


def run_case_b(step_counts, verbose=True, bail_seconds=10.0):
    """Run case (b) for each K in step_counts.

    Stops early if lambdify_np time exceeds bail_seconds.
    """
    rows = []
    for k in step_counts:
        if verbose:
            print(f"  K={k:3d} limit steps...", end=" ", flush=True)

        row = bench_chain_limit_flowprog(k)
        rows.append(row)

        if verbose:
            print(
                f"build={row['build']*1e3:6.2f}ms  compile={row['compile']*1e3:6.2f}ms  "
                f"lambdify np={row['lambdify_np']*1e3:7.2f}ms math={row['lambdify_math']*1e3:7.2f}ms  "
                f"eval(math)={row['eval_math']*1e6:6.1f}µs"
            )

        if row["lambdify_np"] >= bail_seconds:
            if verbose:
                print(
                    f"  (stopping: lambdify(numpy)={row['lambdify_np']:.1f}s "
                    f">= bail threshold {bail_seconds}s)"
                )
            break

    return rows


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

SIZES_FULL = [5, 10, 20, 50, 100, 200]
SIZES_QUICK = [5, 10, 20]
STEPS_FULL = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16]
STEPS_QUICK = [1, 2, 3, 4, 5]
BAIL_SECONDS = 10.0


def main():
    parser = argparse.ArgumentParser(description="Flowprog performance benchmark")
    parser.add_argument("--case", choices=["a", "b"], help="Run only this case")
    parser.add_argument("--quick", action="store_true", help="Smaller sizes for quick smoke-test")
    parser.add_argument("--bail", type=float, default=BAIL_SECONDS,
                        help=f"Stop case (b) when lambdify exceeds this many seconds (default {BAIL_SECONDS})")
    args = parser.parse_args()

    sizes = SIZES_QUICK if args.quick else SIZES_FULL
    steps = STEPS_QUICK if args.quick else STEPS_FULL

    run_a = args.case in (None, "a")
    run_b = args.case in (None, "b")

    bw2data, _ = _try_import_bw()

    print("=" * 78)
    print("Flowprog performance benchmark")
    print("=" * 78)

    if run_a:
        print()
        print("Case (a): Plain pull_production — chain and fan variants")
        print("-" * 78)
        print("Flowprog: build | compile | lambdify(np/math) | eval(np/math)")
        print("  lambdify(math): pure-Python function, lower overhead for scalar eval")
        print("Scipy baseline:  matrix_build | matrix_solve")
        if bw2data:
            print("Brightway:       bw_db_setup | bw_lci_solve")
        else:
            print("(brightway not installed — scipy matrix solver used as comparison)")
        print()
        run_case_a(sizes)

    if run_b:
        print()
        print("Case (b): Chain-of-processes limit model — K limit steps")
        print("-" * 78)
        print("Chain: P1(output←int1) → P2(int1←int2) → … → Pfinal(intK)")
        print("Each step adds a Piecewise limit on the next process's output,")
        print("referencing accumulated production deficit from prior steps.")
        print("Flowprog: build | compile | lambdify(np/math) | eval(math)")
        print("  eval uses math module: SymPy's numpy codegen miscompiles nested")
        print("  Piecewise/ITE nodes for scalar inputs.")
        print(f"(auto-stops when lambdify(numpy) > {args.bail:.0f}s)")
        print()
        run_case_b(steps, bail_seconds=args.bail)

    print()
    print("Done.")


if __name__ == "__main__":
    main()
