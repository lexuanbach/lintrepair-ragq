"""Run the full LintRepair-RAG-Q pilot.

Detection (RQ1) is computed once with the Sonnet detector.
Repair (RQ2-RQ4) is evaluated for several configurations and two repair
backbones, holding the detector findings fixed:

  B0  plain-LLM         (no findings, no retrieval, no validator)
  B1  RAG-LLM           (findings + retrieval, single shot, no validator)
  B2  full, naive       (B1 + validator loop, findings stated as-is)
  B3  full, debiased     (B2 with an intent-preserving, non-anchoring prompt)

Backbones: claude-sonnet-4-6 (B0..B3) and claude-haiku-4-5 (B0,B2,B3) to test
when the validator loop earns its keep on a weaker model.

All LLM calls are disk-cached; re-runs reproduce identical correctness metrics.
Latency is wall-clock during the live run.
"""
from __future__ import annotations

import ast
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from lrq import repair as R
from lrq.detectors import detect
from lrq.harness import run_script, results_match
from lrq.stats import wilson, paired_bootstrap_diff, mcnemar
from lrq.llm import usage_report

RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)

PASS_K = 3
MAX_ITERS = 3
WORKERS = 5
SONNET = "claude-sonnet-4-6"
HAIKU = "claude-haiku-4-5-20251001"

# method spec: fn, debias, samples per model
SUITE = {
    SONNET: {
        "B0": ("plain", False, PASS_K),
        "B1": ("rag", False, 1),
        "B2": ("full", False, PASS_K),
        "B3": ("full", True, PASS_K),
    },
    HAIKU: {
        "B0": ("plain", False, 1),
        "B2": ("full", False, 1),
        "B3": ("full", True, 1),
    },
}


def accept_set(inst) -> set:
    if inst["kind"] == "clean":
        return set()
    s = {inst["category"]}
    if inst["category"] == "deprecated-execute-api":
        s.add("removed-aer-import")
    return s


def compiles(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def _run_method(fn, debias, src, findings, model, sample):
    if fn == "plain":
        return R.plain_repair(src, sample=sample, model=model)
    if fn == "rag":
        return R.rag_repair(src, findings, sample=sample, debias=debias, model=model)
    return R.full_repair(src, findings, run_script, sample=sample,
                         max_iters=MAX_ITERS, debias=debias, model=model)


def process(inst) -> dict:
    src = inst["buggy"]
    ref = inst["reference_result"]
    rec = {"id": inst["id"], "category": inst["category"], "kind": inst["kind"]}

    # detection with the Sonnet detector
    static_cats = sorted({f.category for f in detect(src)})
    llm_cats = sorted({f.category for f in R.llm_detect(src, model=SONNET)})
    rec["detection"] = {"static": static_cats, "llm": llm_cats,
                        "union": sorted(set(static_cats) | set(llm_cats))}
    if inst["kind"] == "clean":
        return rec

    findings = R.all_findings(src)  # static + sonnet-llm union (cached)

    def score(code):
        out = run_script(code)
        ran = out.status == "ok"
        sem = results_match(ref, out.result)[0] if ran else False
        return {"compile": compiles(code), "ran": ran, "sem": bool(sem)}

    rec["repair"] = {}
    for model, methods in SUITE.items():
        mres = {}
        for name, (fn, debias, samples) in methods.items():
            samp = []
            for k in range(samples):
                r = _run_method(fn, debias, src, findings, model, k)
                s = score(r.code)
                s.update(latency=r.latency, calls=r.llm_calls, iters=getattr(r, "iters", 1))
                samp.append(s)
            mres[name] = {"samples": samp, "pass1": samp[0],
                          "passk": any(x["sem"] for x in samp)}
        rec["repair"][model] = mres
    return rec


def det_metrics(records, bench, key):
    tp = fp = fn = hits = nb = 0
    for r, inst in zip(records, bench):
        acc = accept_set(inst)
        found = set(r["detection"][key])
        tp += len(found & acc)
        fp += len(found - acc)
        if inst["kind"] != "clean":
            nb += 1
            if inst["category"] in found:
                hits += 1
            else:
                fn += 1
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec_ = hits / nb if nb else 0.0
    f1 = 2 * prec * rec_ / (prec + rec_) if (prec + rec_) else 0.0
    return {"precision": prec, "recall": rec_, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def main():
    bench = json.load(open(os.path.join(ROOT, "benchmark", "benchmark.json")))
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        records = list(ex.map(process, bench))
    wall = time.time() - t0
    records.sort(key=lambda r: r["id"])
    order = {r["id"]: r for r in records}
    bench_sorted = sorted(bench, key=lambda b: b["id"])
    records = [order[b["id"]] for b in bench_sorted]
    bench = bench_sorted

    cats = sorted({b["category"] for b in bench if b["kind"] != "clean"})
    buggy = [(r, b) for r, b in zip(records, bench) if b["kind"] != "clean"]

    detection = {k: det_metrics(records, bench, k) for k in ("static", "llm", "union")}
    det_by_cat = {}
    for cat in cats:
        items = [(r, b) for r, b in buggy if b["category"] == cat]
        det_by_cat[cat] = {key: [sum(1 for r, b in items if cat in set(r["detection"][key])), len(items)]
                           for key in ("static", "llm", "union")}

    def col(model, method, field, passk=False):
        out = []
        for r, b in buggy:
            m = r["repair"].get(model, {}).get(method)
            if not m:
                continue
            out.append(1 if (m["passk"] if passk else m["pass1"][field]) else 0)
        return out

    repair = {}
    for model, methods in SUITE.items():
        repair[model] = {}
        for name, (fn, debias, samples) in methods.items():
            s = col(model, name, "sem")
            n = len(s)
            p, lo, hi = wilson(sum(s), n)
            ms = [r["repair"][model][name] for r, b in buggy]
            entry = {
                "n": n,
                "pass1_sem": [sum(s), n, p, lo, hi],
                "compile_rate": sum(col(model, name, "compile")) / n,
                "exec_rate": sum(col(model, name, "ran")) / n,
                "mean_latency": sum(m["pass1"]["latency"] for m in ms) / n,
                "mean_calls": sum(m["pass1"]["calls"] for m in ms) / n,
                "mean_iters": sum(m["pass1"].get("iters", 1) for m in ms) / n,
            }
            if samples > 1:
                pk = col(model, name, "sem", passk=True)
                entry["passk_sem"] = [sum(pk), n] + list(wilson(sum(pk), n)) + [samples]
            repair[model][name] = entry

    # paired comparisons (Sonnet) on pass@1 semantic success
    paired = {}
    sb = {m: col(SONNET, m, "sem") for m in SUITE[SONNET]}
    for a, b in [("B2", "B0"), ("B3", "B0"), ("B3", "B2"), ("B1", "B0"), ("B3", "B1")]:
        d, lo, hi = paired_bootstrap_diff(sb[a], sb[b])
        mc = mcnemar(sb[b], sb[a])
        paired[f"{a}_vs_{b}"] = {"diff": d, "lo": lo, "hi": hi,
                                 "mcnemar_p": mc[2], "wins": mc[0], "losses": mc[1]}
    # Haiku validator effect
    hb = {m: col(HAIKU, m, "sem") for m in SUITE[HAIKU]}
    for a, b in [("B2", "B0"), ("B3", "B0"), ("B3", "B2")]:
        d, lo, hi = paired_bootstrap_diff(hb[a], hb[b])
        mc = mcnemar(hb[b], hb[a])
        paired[f"HAIKU_{a}_vs_{b}"] = {"diff": d, "lo": lo, "hi": hi,
                                       "mcnemar_p": mc[2], "wins": mc[0], "losses": mc[1]}

    repair_by_cat = {}
    bm = {r["id"]: r for r, b in buggy}
    for cat in cats:
        ids = [b["id"] for r, b in buggy if b["category"] == cat]
        repair_by_cat[cat] = {}
        for model in SUITE:
            for name in SUITE[model]:
                hits = sum(1 for _id in ids if bm[_id]["repair"][model][name]["pass1"]["sem"])
                repair_by_cat[cat][f"{('S' if model==SONNET else 'H')}:{name}"] = [hits, len(ids)]

    summary = {
        "n_instances": len(bench), "n_buggy": len(buggy),
        "n_clean": len(bench) - len(buggy), "n_categories": len(cats),
        "wall_seconds": wall, "usage": usage_report(),
        "models": {"sonnet": SONNET, "haiku": HAIKU},
        "detection": detection, "detection_by_category": det_by_cat,
        "repair": repair, "repair_by_category": repair_by_cat, "paired": paired,
        "config": {"pass_k": PASS_K, "max_iters": MAX_ITERS, "shots": 8192,
                   "tvd_tol": 0.08, "fid_tol": 0.99},
    }
    json.dump(records, open(os.path.join(RESULTS, "records.json"), "w"), indent=2)
    json.dump(summary, open(os.path.join(RESULTS, "summary.json"), "w"), indent=2)

    # console
    print(f"\n=== {summary['n_buggy']} buggy + {summary['n_clean']} clean, "
          f"{summary['n_categories']} cats, wall={wall:.0f}s ===")
    print("USAGE:", summary["usage"])
    print("\nDETECTION (Sonnet) P / R / F1:")
    for k in ("static", "llm", "union"):
        d = detection[k]
        print(f"  {k:7s} P={d['precision']:.3f} R={d['recall']:.3f} F1={d['f1']:.3f} (tp={d['tp']} fp={d['fp']} fn={d['fn']})")
    for model, tag in [(SONNET, "SONNET"), (HAIKU, "HAIKU")]:
        print(f"\nREPAIR [{tag}] pass@1 semantic:")
        for name in SUITE[model]:
            e = repair[model][name]
            k, n, p, lo, hi = e["pass1_sem"]
            pk = f" pass@{e['passk_sem'][-1]}={e['passk_sem'][4]:.3f}" if "passk_sem" in e else ""
            print(f"  {name}: {p:.3f}[{lo:.3f},{hi:.3f}] compile={e['compile_rate']:.3f} "
                  f"exec={e['exec_rate']:.3f} lat={e['mean_latency']:.1f}s calls={e['mean_calls']:.2f} "
                  f"iters={e['mean_iters']:.2f}{pk}")
    print("\nPAIRED (pass@1 sem; diff [lo,hi], McNemar p, wins/losses):")
    for k, v in paired.items():
        print(f"  {k:18s} {v['diff']:+.3f} [{v['lo']:+.3f},{v['hi']:+.3f}] p={v['mcnemar_p']:.3f} w/l={v['wins']}/{v['losses']}")
    print("\nREPAIR by category:")
    hdr = list(next(iter(repair_by_cat.values())).keys())
    print("  " + " " * 26 + " ".join(f"{h:>7s}" for h in hdr))
    for cat, row in repair_by_cat.items():
        print(f"  {cat:26s}" + " ".join(f"{row[h][0]:>3}/{row[h][1]:<3}" for h in hdr))
    print(f"\nwrote {RESULTS}/summary.json")


if __name__ == "__main__":
    main()
