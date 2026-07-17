"""Powered interference test on the 115 mutation-generated silent bugs (approach #2).
Runs B0 (plain), B2 (grounded: static findings -- empty on these silent bugs, so the
prompt says 'no structural defects' -- + retrieval + validator), and B0filler
(length/retrieval-matched control) across weak->strong models. Oracle: repaired
RESULT matches the base program's reference (results_match). This re-tests the
strong-model interference at n=115 vs the n=9 dissection. Writes
results/mutation_crossover.json.
"""
from __future__ import annotations
import json, os, sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT)
from lrq import repair as R
from lrq.detectors import detect
from lrq.retrieval import TfidfRetriever
from lrq.repair import extract_code, _REPAIR_SYS
from lrq.harness import run_script, results_match
from lrq.llm import complete, usage_report

BUGS = json.load(open(os.path.join(HERE, "results", "mutation_dataset.json")))["silent_bugs"]
RETR = TfidfRetriever()
MODELS = sys.argv[1:] or ["gpt-4o-mini", "claude-haiku-4-5", "claude-sonnet-4-6"]


def score(code, ref):
    o = run_script(code); return bool(results_match(ref, o.result)[0]) if o.status == "ok" else False


def b0filler(buggy, model):
    docs = RETR.retrieve(buggy, k=4); ctx = "\n".join(f"- [{d.tag}] {d.text}" for d in docs)
    p = (f"The following Qiskit program is buggy. Fix it.\n\nGeneral background reference "
         f"(not specific to this program):\n{ctx}\n\n```python\n{buggy}\n```")
    return extract_code(complete(_REPAIR_SYS, p, temperature=0.0, max_tokens=1500, model=model))


def one(bug, model):
    src = bug["buggy"]; ref = bug["reference"]; f = detect(src)
    return {
        "B0": score(R.plain_repair(src, model=model).code, ref),
        "B2": score(R.full_repair(src, f, run_script, max_iters=3, debias=False, model=model).code, ref),
        "B0filler": score(b0filler(src, model), ref),
    }


OUT = os.path.join(HERE, "results", "mutation_crossover.json")


def main():
    n = len(BUGS)
    out = json.load(open(OUT)) if os.path.exists(OUT) else {}   # additive: keep prior models
    print(f"powered interference test: {n} mutation silent bugs x {len(MODELS)} models\n")
    for model in MODELS:
        workers = 2 if ":" in model else 5                      # local (ollama) -> fewer concurrent
        with ThreadPoolExecutor(max_workers=workers) as ex:
            recs = list(ex.map(lambda b: one(b, model), BUGS))
        agg = {c: sum(1 for r in recs if r[c]) for c in ["B0", "B2", "B0filler"]}
        out[model] = {**agg, "n": n}
        print(f"  {model:22s} B0={agg['B0']}/{n} B2={agg['B2']}/{n} B0filler={agg['B0filler']}/{n}  "
              f"| B2-B0={agg['B2']-agg['B0']:+d}  B0filler-B0={agg['B0filler']-agg['B0']:+d}")
        print("    usage:", usage_report(), flush=True)
        json.dump(out, open(OUT, "w"), indent=2)


if __name__ == "__main__":
    main()
