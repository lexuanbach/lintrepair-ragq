"""Reviewer Q3: oracle-threshold + seed sensitivity. For each of the 32 defective
programs we execute the human FIX (should pass) and the BUGGY version (should fail)
and record the raw oracle metric vs the reference: TVD over counts (lower=better,
default tol 0.08) or statevector fidelity (higher=better, default tol 0.99). We show
(a) pass/fail classification is stable across a wide threshold band, and (b) shot-seed
jitter is far inside the margin. Writes results/oracle_sensitivity.json.
"""
from __future__ import annotations
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lrq.harness import run_script, results_match

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
BENCH = [b for b in json.load(open(os.path.join(ROOT, "benchmark", "benchmark.json"))) if b["kind"] != "clean"]


def metric(code, ref):
    """(type, value) where value is TVD (counts) or fidelity (state); None if no run."""
    o = run_script(code)
    if o.status != "ok" or o.result is None or o.result.get("type") != ref.get("type"):
        return ref.get("type"), None
    _, v = results_match(ref, o.result)
    return ref["type"], v


def stats(xs):
    xs = sorted(xs); n = len(xs)
    return None if n == 0 else {"n": n, "min": round(xs[0], 4),
            "median": round(xs[n // 2], 4), "max": round(xs[-1], 4)}


def main():
    counts_fixed, counts_buggy, state_fixed, state_buggy = [], [], [], []
    crash = 0
    for b in BENCH:
        ref = b["reference_result"]
        t, vf = metric(b["fixed"], ref)
        _, vb = metric(b["buggy"], ref)
        if t == "counts":
            if vf is not None: counts_fixed.append(vf)
            if vb is not None: counts_buggy.append(vb)
            else: crash += 1
        else:
            if vf is not None: state_fixed.append(vf)
            if vb is not None: state_buggy.append(vb)
            else: crash += 1

    # TVD sweep (counts): pass iff TVD <= tol
    tvd_sweep = {}
    for tol in (0.02, 0.04, 0.08, 0.12, 0.16, 0.20):
        tvd_sweep[tol] = {"fixed_pass": sum(1 for d in counts_fixed if d <= tol),
                          "fixed_total": len(counts_fixed),
                          "buggy_false_accept": sum(1 for d in counts_buggy if d <= tol),
                          "buggy_run_total": len(counts_buggy)}
    # fidelity sweep (state): pass iff fidelity >= tol
    fid_sweep = {}
    for tol in (0.90, 0.95, 0.99, 0.999):
        fid_sweep[tol] = {"fixed_pass": sum(1 for f in state_fixed if f >= tol),
                          "fixed_total": len(state_fixed),
                          "buggy_false_accept": sum(1 for f in state_buggy if f >= tol),
                          "buggy_run_total": len(state_buggy)}

    # shot-seed jitter: re-run each counts fix 5x, record max TVD-to-reference
    jitter = []
    for b in BENCH:
        if b["reference_result"].get("type") != "counts":
            continue
        vs = [metric(b["fixed"], b["reference_result"])[1] for _ in range(5)]
        vs = [v for v in vs if v is not None]
        if vs: jitter.append(max(vs))

    out = {"counts_fixed_tvd": stats(counts_fixed), "counts_buggy_tvd": stats(counts_buggy),
           "state_fixed_fid": stats(state_fixed), "state_buggy_fid": stats(state_buggy),
           "buggy_crash_nonrun": crash, "tvd_sweep": tvd_sweep, "fid_sweep": fid_sweep,
           "shot_seed_jitter_max_tvd": stats(jitter)}
    json.dump(out, open(os.path.join(HERE, "results", "oracle_sensitivity.json"), "w"), indent=2)

    print("Oracle threshold + seed sensitivity (Q3)\n")
    print(f"  counts: fixed TVD {out['counts_fixed_tvd']}")
    print(f"          buggy TVD {out['counts_buggy_tvd']}  ({crash} buggy crash->trivial fail)")
    print(f"  state:  fixed fidelity {out['state_fixed_fid']}")
    print(f"          buggy fidelity {out['state_buggy_fid']}")
    print(f"  shot-seed jitter (max TVD over 5 reruns of each fix): {out['shot_seed_jitter_max_tvd']}")
    print("\n  TVD sweep (tol -> fixed pass / buggy false-accept):")
    for tol, s in tvd_sweep.items():
        print(f"    tol={tol:.2f}: fixed {s['fixed_pass']}/{s['fixed_total']}, "
              f"buggy false-accept {s['buggy_false_accept']}/{s['buggy_run_total']}")
    print("  fidelity sweep (tol -> fixed pass / buggy false-accept):")
    for tol, s in fid_sweep.items():
        print(f"    tol={tol:.3f}: fixed {s['fixed_pass']}/{s['fixed_total']}, "
              f"buggy false-accept {s['buggy_false_accept']}/{s['buggy_run_total']}")


if __name__ == "__main__":
    main()
