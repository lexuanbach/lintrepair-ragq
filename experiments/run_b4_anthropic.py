"""Fill the missing B4 (generic-debiased) cell for the two frozen Anthropic
models (Sonnet 4.6, Haiku 4.5). B4 postdated the frozen pilot, so it is run here
with the SAME run_one logic as the cross-model study (findings reconstructed from
the frozen records; only the repair backbone + debias='generic' vary). Writes
results/b4_anthropic.json = {model: {id: {compile,ran,sem,latency,iters}}}.
Keeps records_FROZEN.json untouched; aggregate_models.py merges this supplement.
"""
from __future__ import annotations
import json, os, sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_models as RM
from lrq.llm import usage_report

MODELS = ["claude-sonnet-4-6", "claude-haiku-4-5-20251001"]
OUT = os.path.join(HERE, "results", "b4_anthropic.json")


def main():
    out = {}
    for model in MODELS:
        print(f"\n=== {model}  B4 (generic-debiased) over {len(RM.BENCH)} programs ===", flush=True)
        with ThreadPoolExecutor(max_workers=4) as ex:
            recs = list(ex.map(lambda inst: RM.run_one(inst, model, ["B4"]), RM.BENCH))
        per = {r["id"]: r["B4"] for r in recs}
        out[model] = per
        k = sum(1 for v in per.values() if v["sem"])
        print(f"  B4 pass@1 (sem): {k}/{len(per)} = {k/len(per):.3f}   usage={usage_report()}", flush=True)
        json.dump(out, open(OUT, "w"), indent=2)
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
