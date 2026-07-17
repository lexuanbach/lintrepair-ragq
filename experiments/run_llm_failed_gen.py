"""Approach #3: harvest LLM-FAILED generations as natural bugs. We give QuanBench+'s
Cirq task prompts to a weak model, run each generated circuit, and compare its
basis-state distribution to the task's canonical_output (an SDK-agnostic oracle).
Generations that fail are REAL bugs at the model's own difficulty frontier -- by
construction in the sweet-spot where interference could appear. Writes
results/llm_failed_dataset.json (each failure = a buggy program + its reference).
"""
from __future__ import annotations
import json, os, sys, re, contextlib, io
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT)
from lrq.llm import complete, usage_report
import cirq, numpy as np

# External QuanBench+ prompt set (auxiliary; result frozen in results/).
QB = os.environ.get("QUANBENCH_DIR", "/tmp/quanbench-plus")
TASKS = [json.loads(l) for l in open(f"{QB}/prompts/cirq.jsonl")]
CANON = {c["task_id"]: c["canonical_output"] for c in json.load(open(f"{QB}/canonical_results/canonical_solutions.json"))}
MODEL = "gpt-4o-mini"; TOL = 0.10; SHOTS = 2000


def extract(txt):
    m = re.findall(r"```(?:python)?\s*\n(.*?)```", txt, re.DOTALL)
    return (max(m, key=len) if m else txt).strip()


def dist_of(code, entry, nstates):
    ns = {"cirq": cirq, "np": np}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(code, ns)
        circ = ns[entry]()
        res = cirq.Simulator(seed=1).run(circ, repetitions=SHOTS)
    rows = res.measurements["result"]
    d = [0.0] * nstates
    for r in rows:
        idx = int("".join(str(int(b)) for b in r), 2)
        if idx < nstates:
            d[idx] += 1
    s = sum(d) or 1
    return [x / s for x in d]


def tvd(a, b):
    n = max(len(a), len(b)); a = a + [0] * (n - len(a)); b = b + [0] * (n - len(b))
    return 0.5 * sum(abs(x - y) for x, y in zip(a, b))


def gen_and_check(task):
    tid = task["task_id"]; canon = CANON.get(tid)
    if canon is None:
        return None
    prompt = task["complete_prompt"] + "\n\nReturn the COMPLETE function in one ```python block."
    code = extract(complete("You are a Cirq expert. Complete the function exactly as specified.",
                            prompt, temperature=0.0, max_tokens=900, model=MODEL))
    try:
        d = dist_of(code, task["entry_point"], len(canon))
        err = tvd(d, canon)
        return {"task_id": tid, "entry": task["entry_point"], "status": "ran",
                "tvd_vs_canonical": round(err, 3), "is_bug": err > TOL,
                "buggy": code, "reference": canon}
    except Exception as e:
        return {"task_id": tid, "entry": task["entry_point"], "status": "crash",
                "error": f"{type(e).__name__}: {str(e)[:80]}", "is_bug": True,
                "buggy": code, "reference": canon}


def main():
    subset = TASKS[:20]
    with ThreadPoolExecutor(max_workers=4) as ex:
        rows = [r for r in ex.map(gen_and_check, subset) if r]
    bugs = [r for r in rows if r["is_bug"]]
    crash = sum(1 for r in bugs if r["status"] == "crash")
    wrong = sum(1 for r in bugs if r["status"] == "ran")
    print(f"LLM-failed harvest ({MODEL}, {len(rows)} Cirq tasks):")
    print(f"  failed generations (bugs): {len(bugs)}/{len(rows)}  (wrong-output {wrong}, crash {crash})")
    print(f"  -> usable as a {len(bugs)}-program natural-bug set at the model's frontier")
    print("  usage:", usage_report())
    json.dump({"summary": {"tasks": len(rows), "bugs": len(bugs), "wrong_output": wrong, "crash": crash},
               "bugs": bugs}, open(os.path.join(HERE, "results", "llm_failed_dataset.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
