"""Cross-model repair study: extend the pilot to additional LLM backbones.

Models added: gpt-4o-mini (OpenAI), gemini-2.5-flash-lite (Google),
qwen2.5:7b and llama3.1:8b (local Ollama, 7-8B open weights). Sonnet 4.6 and
Haiku 4.5 results come from the frozen run (summary_FROZEN.json).

Detection findings are reconstructed from the frozen records (static + the
Sonnet LLM detector's categories) so NO Anthropic API call is needed -- only
the repair backbone varies, exactly as in the main study. Results are written
to results/models_extra.json and never overwrite the frozen files.

Usage: python experiments/run_models.py "gpt-4o-mini" "gemini-2.5-flash-lite" ...
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
from lrq.detectors import detect, Finding
from lrq.harness import run_script, results_match
from lrq.stats import wilson, paired_bootstrap_diff, mcnemar
from lrq.llm import usage_report, provider_of

RESULTS = os.path.join(HERE, "results")
BENCH = [b for b in json.load(open(os.path.join(ROOT, "benchmark", "benchmark.json")))
         if b["kind"] != "clean"]  # repair the 32 defective programs only
FROZEN = {r["id"]: r for r in json.load(open(os.path.join(RESULTS, "records_FROZEN.json")))}
MAX_ITERS = 3

# configs to run per model (B1 only for API models to save local compute)
# B4 = generic (non-leaking) debiasing -- isolates anchoring-removal from family naming.
CONFIGS_FULL = ["B0", "B1", "B2", "B3", "B4"]
CONFIGS_OLLAMA = ["B0", "B2", "B3", "B4"]


def compiles(code):
    try:
        ast.parse(code); return True
    except SyntaxError:
        return False


def findings_for(inst):
    """Reconstruct union findings (static + frozen Sonnet-LLM) without any API."""
    static = detect(inst["buggy"])
    cats = {f.category for f in static}
    out = list(static)
    for cat in FROZEN[inst["id"]]["detection"]["union"]:
        if cat not in cats:
            out.append(Finding(cat, 0, "(LLM-detected)"))
    return out


def run_one(inst, model, configs):
    src = inst["buggy"]; ref = inst["reference_result"]
    f = findings_for(inst)
    rec = {"id": inst["id"], "category": inst["category"]}
    def score(code):
        out = run_script(code)
        ran = out.status == "ok"
        sem = results_match(ref, out.result)[0] if ran else False
        return {"compile": compiles(code), "ran": ran, "sem": bool(sem)}
    for cfg in configs:
        if cfg == "B0":
            r = R.plain_repair(src, sample=0, model=model)
        elif cfg == "B1":
            r = R.rag_repair(src, f, sample=0, debias=False, model=model)
        elif cfg == "B2":
            r = R.full_repair(src, f, run_script, sample=0, max_iters=MAX_ITERS, debias=False, model=model)
        elif cfg == "B3":
            r = R.full_repair(src, f, run_script, sample=0, max_iters=MAX_ITERS, debias="named", model=model)
        else:  # B4 generic debiasing (no family names)
            r = R.full_repair(src, f, run_script, sample=0, max_iters=MAX_ITERS, debias="generic", model=model)
        s = score(r.code); s.update(latency=r.latency, iters=getattr(r, "iters", 1))
        rec[cfg] = s
    return rec


def metrics_for(records, configs):
    buggy = [r for r in records]
    out = {"n": len(buggy)}
    def col(cfg, field):
        return [1 if r[cfg][field] else 0 for r in buggy]
    for cfg in configs:
        s = col(cfg, "sem"); n = len(s)
        p, lo, hi = wilson(sum(s), n)
        out[cfg] = {"pass1": [sum(s), n, round(p, 4), round(lo, 4), round(hi, 4)],
                    "compile": round(sum(col(cfg, "compile")) / n, 4),
                    "exec": round(sum(col(cfg, "ran")) / n, 4),
                    "mean_latency": round(sum(r[cfg]["latency"] for r in buggy) / n, 2),
                    "mean_iters": round(sum(r[cfg].get("iters", 1) for r in buggy) / n, 3)}
    out["rho_hat"] = out["B0"]["exec"]  # base-model single-shot run probability
    # paired vs B0
    b0 = col("B0", "sem")
    out["paired_vs_B0"] = {}
    for cfg in configs:
        if cfg == "B0":
            continue
        c = col(cfg, "sem")
        d, lo, hi = paired_bootstrap_diff(c, b0)
        mc = mcnemar(b0, c)
        out["paired_vs_B0"][cfg] = {"diff": round(d, 4), "ci": [round(lo, 4), round(hi, 4)],
                                    "mcnemar_p": round(mc[2], 4), "wins": mc[0], "losses": mc[1]}
    def pair(a, b):
        ca, cb = col(a, "sem"), col(b, "sem")
        d, lo, hi = paired_bootstrap_diff(ca, cb); mc = mcnemar(cb, ca)
        return {"diff": round(d, 4), "ci": [round(lo, 4), round(hi, 4)],
                "mcnemar_p": round(mc[2], 4), "wins": mc[0], "losses": mc[1]}
    for a, b in [("B3", "B2"), ("B4", "B2"), ("B4", "B3")]:
        if a in configs and b in configs:
            out[f"{a}_vs_{b}"] = pair(a, b)
    return out


def main(models):
    out = {}
    try:
        prev = json.load(open(os.path.join(RESULTS, "models_extra.json")))
    except Exception:
        prev = {}
    out.update(prev)
    for model in models:
        prov = provider_of(model)
        configs = CONFIGS_OLLAMA if prov == "ollama" else CONFIGS_FULL
        workers = 2 if prov == "ollama" else 5
        print(f"\n=== {model} ({prov}, {configs}, workers={workers}) ===", flush=True)
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=workers) as ex:
            recs = list(ex.map(lambda inst: run_one(inst, model, configs), BENCH))
        m = metrics_for(recs, configs)
        m["model"] = model; m["provider"] = prov; m["wall_s"] = round(time.time() - t0)
        out[model] = {"metrics": m, "records": recs}
        json.dump(out, open(os.path.join(RESULTS, "models_extra.json"), "w"), indent=2)
        r = m
        line = " ".join(f"{c}={r[c]['pass1'][2]:.3f}({r[c]['pass1'][0]}/{r[c]['pass1'][1]},exec={r[c]['exec']:.2f})" for c in configs)
        print(f"  rho_hat={r['rho_hat']:.3f}  {line}")
        if "B3_vs_B2" in r:
            print(f"  B3_vs_B2 diff={r['B3_vs_B2']['diff']:+.3f} CI{r['B3_vs_B2']['ci']} w/l={r['B3_vs_B2']['wins']}/{r['B3_vs_B2']['losses']}")
        print(f"  wall={m['wall_s']}s usage={usage_report()}", flush=True)
    print("\nwrote", os.path.join(RESULTS, "models_extra.json"))


if __name__ == "__main__":
    models = sys.argv[1:] or ["gpt-4o-mini", "gemini-2.5-flash-lite"]
    main(models)
