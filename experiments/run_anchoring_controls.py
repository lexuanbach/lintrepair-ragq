"""Anchoring causal controls on the two anchored models (Sonnet, Haiku),
addressing reviewer Q2/Q8: is the anchoring really the analyser's cleanliness
framing, or a confound (prompt length / retrieval distraction)?

Conditions over the 32 defective programs (vs existing B0 and B2):
  B0filler  -- B0 plus the retrieved KB docs relabelled as neutral "general
               background" (NO findings, NO cleanliness claim, NO validator):
               length- and retrieval-content-matched to grounding. If this stays
               near B0, prompt length / retrieval content alone do not anchor.
  B2irrel   -- naive grounding (findings + "no defects" when silent + validator)
               but with IRRELEVANT retrieved docs (fixed unrelated query). If this
               anchors like B2, the harm is the findings framing, not retrieval.

Writes results/anchoring_controls.json. Anthropic (cached after first run).
"""
from __future__ import annotations
import json, os, sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from lrq.detectors import detect, Finding
from lrq.retrieval import TfidfRetriever
from lrq.repair import _findings_block, extract_code, _REPAIR_SYS
from lrq.harness import run_script, results_match
from lrq.llm import complete, last_latency, usage_report

BENCH = [b for b in json.load(open(os.path.join(ROOT, "benchmark", "benchmark.json"))) if b["kind"] != "clean"]
FROZEN = {r["id"]: r for r in json.load(open(os.path.join(HERE, "results", "records_FROZEN.json")))}
RETR = TfidfRetriever()
MODELS = ["claude-sonnet-4-6", "claude-haiku-4-5-20251001"]
OUT = os.path.join(HERE, "results", "anchoring_controls.json")
IRRELEVANT_QUERY = "matplotlib circuit diagram drawing styles colors latex figure layout"


def union_findings(inst):
    static = detect(inst["buggy"]); cats = {f.category for f in static}; out = list(static)
    for cat in FROZEN[inst["id"]]["detection"]["union"]:
        if cat not in cats:
            out.append(Finding(cat, 0, "(LLM-detected)"))
    return out


def correct(code, ref):
    o = run_script(code); return bool(results_match(ref, o.result)[0]) if o.status == "ok" else False


def b0_filler(inst, model):
    """B0 + retrieved docs as neutral background (length/retrieval matched)."""
    docs = RETR.retrieve(inst["buggy"], k=4)
    ctx = "\n".join(f"- [{d.tag}] {d.text}" for d in docs)
    prompt = (f"The following Qiskit program is buggy. Fix it.\n\nGeneral background "
              f"reference (not specific to this program):\n{ctx}\n\n```python\n{inst['buggy']}\n```")
    return extract_code(complete(_REPAIR_SYS, prompt, temperature=0.0, max_tokens=1500, model=model))


def b2_irrelevant(inst, model):
    """Naive grounding (findings + validator) but with IRRELEVANT retrieved docs."""
    findings = union_findings(inst)
    docs = RETR.retrieve(IRRELEVANT_QUERY, k=4)        # irrelevant on purpose
    ctx = "\n".join(f"- [{d.tag}] {d.text}" for d in docs)
    history, code = "", inst["buggy"]
    for it in range(3):
        block = _findings_block(findings, False)        # naive: "no defects" when silent
        prompt = (f"Repair this Qiskit program so it runs correctly on Qiskit 2.x.\n\n"
                  f"Detected issues:\n{block}\n\nRelevant knowledge (retrieved):\n{ctx}\n{history}"
                  f"```python\n{inst['buggy']}\n```\n\nReturn the complete corrected program "
                  "in one ```python block, then one line \"EXPLANATION:\".")
        code = extract_code(complete(_REPAIR_SYS, prompt, temperature=0.0, max_tokens=1600, sample=it, model=model))
        o = run_script(code)
        if o.status == "ok":
            break
        history = f"\nYour previous attempt failed validation with this error:\n{o.error}\nFix it.\n"
    return code


def main():
    out = {}
    for model in MODELS:
        print(f"\n=== {model} anchoring controls ===", flush=True)
        def one(inst):
            ref = inst["reference_result"]
            return {"id": inst["id"], "category": inst["category"], "kind": inst["kind"],
                    "B0filler": correct(b0_filler(inst, model), ref),
                    "B2irrel": correct(b2_irrelevant(inst, model), ref)}
        with ThreadPoolExecutor(max_workers=4) as ex:
            rows = list(ex.map(one, BENCH))
        out[model] = rows
        for cfg in ["B0filler", "B2irrel"]:
            k = sum(1 for r in rows if r[cfg]); ks = sum(1 for r in rows if r["kind"] == "semantic" and r[cfg])
            print(f"  {cfg:9s} {k}/32  (silent {ks}/{sum(1 for r in rows if r['kind']=='semantic')})")
        print("  usage:", usage_report(), flush=True)
        json.dump(out, open(OUT, "w"), indent=2)
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
