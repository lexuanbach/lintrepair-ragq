"""Detector-sensitivity controls on the two anchored models (Sonnet, Haiku),
addressing the reviewer question: does the anchoring come from the static
analyser's blindness, the LLM detector's phrasing, or the cleanliness assertion?

Conditions (all are B2-style: retrieval + validator loop), vs the existing B2
(union findings, naive prompt) already in the frozen records:
  B2static  -- grounding findings = STATIC detector only (drop the LLM detector)
  B2noassert-- union findings, but no "no structural defects" claim when empty

Writes results/detector_sensitivity.json. Uses Anthropic (cached after first run).
"""
from __future__ import annotations
import json, os, sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from lrq import repair as R
from lrq.detectors import detect, Finding
from lrq.harness import run_script, results_match
from lrq.llm import usage_report

BENCH = [b for b in json.load(open(os.path.join(ROOT, "benchmark", "benchmark.json"))) if b["kind"] != "clean"]
FROZEN = {r["id"]: r for r in json.load(open(os.path.join(HERE, "results", "records_FROZEN.json")))}
MODELS = ["claude-sonnet-4-6", "claude-haiku-4-5-20251001"]
OUT = os.path.join(HERE, "results", "detector_sensitivity.json")


def union_findings(inst):
    static = detect(inst["buggy"]); cats = {f.category for f in static}; out = list(static)
    for cat in FROZEN[inst["id"]]["detection"]["union"]:
        if cat not in cats:
            out.append(Finding(cat, 0, "(LLM-detected)"))
    return out


def score(code, ref):
    o = run_script(code); ran = o.status == "ok"
    return bool(results_match(ref, o.result)[0]) if ran else False


def run_one(inst, model):
    ref = inst["reference_result"]; src = inst["buggy"]
    static = detect(inst["buggy"])
    r_static = R.full_repair(src, static, run_script, max_iters=3, debias=False, model=model)
    r_noass = R.full_repair(src, union_findings(inst), run_script, max_iters=3, debias="noassert", model=model)
    return {"id": inst["id"], "category": inst["category"],
            "B2static": score(r_static.code, ref), "B2noassert": score(r_noass.code, ref)}


def main():
    out = {}
    for model in MODELS:
        print(f"\n=== {model} detector-sensitivity ===", flush=True)
        with ThreadPoolExecutor(max_workers=4) as ex:
            recs = list(ex.map(lambda inst: run_one(inst, model), BENCH))
        out[model] = recs
        for cfg in ["B2static", "B2noassert"]:
            k = sum(1 for r in recs if r[cfg])
            print(f"  {cfg:11s} pass@1 = {k}/{len(recs)} = {k/len(recs):.3f}")
        print("  usage:", usage_report(), flush=True)
        json.dump(out, open(OUT, "w"), indent=2)
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
