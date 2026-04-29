"""
Flowprog performance benchmark.

Usage:
    python benchmarks/benchmark.py              # both cases, all sizes
    python benchmarks/benchmark.py --case a     # plain pull_production only
    python benchmarks/benchmark.py --case b     # limit() steps only
    python benchmarks/benchmark.py --quick      # small sizes for smoke-testing

Case (a): Plain models using pull_production only
──────────────────────────────────────────────────
Linear chain of N processes: P0→O0←P1→O1←…←P(N-1)→O(N-1).
Each process Pi produces Oi and consumes O(i-1); coefficients S=1, U=0.8.
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

Case (b): Models using limit() with increasing number of steps
───────────────────────────────────────────────────────────────
A single "Supply" process produces "product". We add:
  step 0        : base demand (no limit)
  steps 1..K    : additional demand, each limited by Supply's cumulative output

Each limit() step adds a Piecewise layer that references the accumulated
state from all prior steps, so the expression tree grows exponentially
with K. This tests compile/lambdify performance as branch count explodes.

Same four flowprog phases timed; no brightway comparison (limits have no
direct matrix-solver equivalent).
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
# Case (a): plain pull_production
# ══════════════════════════════════════════════════════════════════════════════

def _make_chain(n: int):
    """Linear chain of N processes + symbol-keyed recipe dict."""
    objects = [Object(f"O{i}", MASS, has_market=True) for i in range(n)]
    processes = [
        Process(f"P{i}", produces=[f"O{i}"], consumes=[f"O{i-1}"] if i > 0 else [])
        for i in range(n)
    ]
    # Build recipe using a temporary builder (symbol objects are SymPy-cached,
    # so the same S[i,j] symbols work across builder instances of the same size)
    tmp = ModelBuilder(processes, objects)
    recipe = {tmp.S[i, i]: 1.0 for i in range(n)}
    recipe.update({tmp.U[i - 1, i]: 0.8 for i in range(1, n)})
    return processes, objects, recipe


DEMAND_A = sy.Symbol("demand", positive=True)


def bench_plain_flowprog(n: int) -> dict:
    """Time the flowprog phases for a plain N-process chain.

    Lambdify is timed for both numpy (default) and math modules.
    Eval is timed for both so we can see scalar-eval overhead from numpy.
    """
    processes, objects, recipe = _make_chain(n)

    # ── Phase 1: build ────────────────────────────────────────────────────
    def do_build():
        b = ModelBuilder(processes, objects)
        b.add(b.pull_production(f"O{n - 1}", DEMAND_A))
        return b

    t_build = _tmin(do_build, reps=5)
    builder = do_build()

    # ── Phase 2: compile ─────────────────────────────────────────────────
    # SympyModel.from_steps(): resolves structural symbols and accumulates
    # values; for plain models this is O(N) work.
    t_compile = _tmin(lambda: builder.build(recipe), reps=5)
    model = builder.build(recipe)

    # ── Phase 3a: lambdify (numpy) ───────────────────────────────────────
    # Evaluates all flow expressions symbolically (O(N²) for a chain via
    # to_flows + eval + intermediate substitution) then calls sy.lambdify()
    # with numpy backend.  Suitable for vectorised / array evaluation.
    t_lambdify_np = _tmin(lambda: model.lambdify(), reps=3)
    fn_np = model.lambdify()

    # ── Phase 3b: lambdify (math) ────────────────────────────────────────
    # Same symbolic work; sy.lambdify() with modules='math' generates a pure
    # Python function (no numpy).  Avoids numpy array overhead for scalar
    # inputs and correctly compiles nested Piecewise / ITE nodes.
    t_lambdify_math = _tmin(lambda: model.lambdify(modules="math"), reps=3)
    fn_math = model.lambdify(modules="math")

    # ── Phase 4a: eval (numpy) ───────────────────────────────────────────
    t_eval_np = _tmin(lambda: fn_np({DEMAND_A: 1.0}), reps=5, calls=50)

    # ── Phase 4b: eval (math) ────────────────────────────────────────────
    t_eval_math = _tmin(lambda: fn_math({DEMAND_A: 1.0}), reps=5, calls=50)

    return dict(
        n=n,
        build=t_build,
        compile=t_compile,
        lambdify_np=t_lambdify_np,
        lambdify_math=t_lambdify_math,
        eval_np=t_eval_np,
        eval_math=t_eval_math,
    )


# ── Scipy matrix comparison ───────────────────────────────────────────────────

def bench_plain_scipy(n: int) -> dict:
    """
    Time a scipy sparse matrix solve equivalent to the N-process chain LCI.

    Technosphere matrix A:
      A[i,i] = 1.0   (production)
      A[i-1,i] = -0.8 (consumption of prior output, negative = consumed)

    Phases:
      matrix_build : construct the sparse CSC matrix
      matrix_solve : spsolve(A, demand_vector)
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
    """
    Time brightway25 for an equivalent N-process chain.

    Phases:
      bw_db_setup  : create project + write activities + exchanges to bw2 DB
      bw_lci_solve : LCA({fu: 1}) + lca.lci()

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

    fp_rows = []
    sp_rows = []
    bw_rows = []

    for n in sizes:
        if verbose:
            print(f"  N={n:4d}...", end=" ", flush=True)

        fp = bench_plain_flowprog(n)
        sp = bench_plain_scipy(n)
        bw = bench_plain_bw25(n) if has_bw else None

        fp_rows.append(fp)
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

    return fp_rows, sp_rows, bw_rows


# ══════════════════════════════════════════════════════════════════════════════
# Case (b): limit() with K steps
# ══════════════════════════════════════════════════════════════════════════════

def _make_limit_model(k: int):
    """
    Build a ModelBuilder with K+1 steps (1 base demand + K limited extras).

    Structure:
      Supply  →  product      (S=1.0)

    Steps:
      step 0   : pull_production("product", demand_sym)   — no limit
      step 1..K: pull_production("product", d_k_sym), limited by Y[Supply] ≤ cap_k
                 cap_k = k + 5.0  (numeric, baked into expression)

    With K limit steps the expression for Y[Supply] contains K nested
    Piecewise clauses, so complexity grows exponentially with K.
    """
    processes = [Process("Supply", produces=["product"], consumes=[])]
    objects = [Object("product", MASS, has_market=True)]
    b = ModelBuilder(processes, objects)

    demand_sym = sy.Symbol("demand", positive=True)
    b.add(b.pull_production("product", demand_sym))

    for step in range(1, k + 1):
        d_sym = sy.Symbol(f"d{step}", positive=True)
        cap = float(step + 5)
        act = b.pull_production("product", d_sym)
        b.add(b.limit(act, b.Y[0], cap))

    recipe = {b.S[0, 0]: 1.0}
    return b, recipe


def bench_limit_flowprog(k: int) -> dict:
    """Time the flowprog phases for a model with K limit steps."""

    # ── Phase 1: build ────────────────────────────────────────────────────
    # Includes all pull_production calls + limit() wrapping + add().
    t_build = _tmin(lambda: _make_limit_model(k), reps=5)
    builder, recipe = _make_limit_model(k)

    # ── Phase 2: compile ─────────────────────────────────────────────────
    # SympyModel.from_steps() processes each Limit transformation, resolving
    # structural symbols and creating Piecewise expressions.  Cost grows with
    # the depth of the accumulated expression tree.
    t_compile = _tmin(lambda: builder.build(recipe), reps=3)
    model = builder.build(recipe)

    # ── Phase 3a: lambdify (numpy) ───────────────────────────────────────
    # For K≥3 the generated numpy code contains ITE nodes that numpy's
    # select cannot handle with scalar inputs — eval will fail there.
    # We still time it because it's part of the compile-time story.
    t_lambdify_np = _tmin(lambda: model.lambdify(), reps=3)

    # ── Phase 3b: lambdify (math) ────────────────────────────────────────
    # modules='math' generates pure Python code; Piecewise / ITE compile
    # correctly and there is no numpy overhead for scalar evaluation.
    t_lambdify_math = _tmin(lambda: model.lambdify(modules="math"), reps=3)
    fn_math = model.lambdify(modules="math")

    # ── Phase 4: eval (math) ─────────────────────────────────────────────
    demand_sym = sy.Symbol("demand", positive=True)
    params = {demand_sym: 1.0}
    params.update({sy.Symbol(f"d{s}", positive=True): 0.5 for s in range(1, k + 1)})
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
    """
    Run case (b) for each K in step_counts.

    Stops early if lambdify time exceeds bail_seconds — once expression
    complexity makes lambdify impractical there is no point continuing to
    larger K values.
    """
    rows = []
    for k in step_counts:
        if verbose:
            print(f"  K={k:3d} steps...", end=" ", flush=True)

        row = bench_limit_flowprog(k)
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
                    f"≥ bail threshold {bail_seconds}s)"
                )
            break

    return rows


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

SIZES_FULL = [5, 10, 20, 50, 100, 200]
SIZES_QUICK = [5, 10, 20]
# Case (b) steps: benchmark stops automatically once lambdify exceeds BAIL_SECONDS
STEPS_FULL = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16]
STEPS_QUICK = [1, 2, 3, 4, 5]
BAIL_SECONDS = 10.0   # stop case (b) if a single lambdify call exceeds this


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

    print("=" * 72)
    print("Flowprog performance benchmark")
    print("=" * 72)

    if run_a:
        print()
        print("Case (a): Plain pull_production — comparable to matrix LCI solver")
        print("-" * 72)
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
        print("Case (b): limit() with K steps — expression branch growth")
        print("-" * 72)
        print("Flowprog: build | compile | lambdify(np/math) | eval(math)")
        print("  lambdify(numpy) timed for completeness; eval uses math because")
        print("  SymPy's numpy code-gen miscompiles nested ITE for scalar inputs.")
        print("(no direct matrix-solver equivalent for limit() models)")
        print(f"(auto-stops when lambdify(numpy) > {args.bail:.0f}s)")
        print()
        run_case_b(steps, bail_seconds=args.bail)

    print()
    print("Done.")


if __name__ == "__main__":
    main()
