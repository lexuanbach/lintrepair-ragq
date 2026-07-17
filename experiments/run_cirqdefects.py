"""Cross-SDK transfer test on Cirq-Defects (benchmark/cirq_defects.py). Does the
grounding-context interference on capable models reproduce on Cirq? We run B0
(plain), B2 (grounded: 'no structural defects' + Cirq retrieval + validator), and
B0filler (length/retrieval-matched control) across weak->strong models. Oracle:
RESULT match (TVD<=0.08 for counts, exact for ints) -- the same oracle as
Q-Defects40. Writes results/cirqdefects.json.
"""
from __future__ import annotations
import json, os, sys, re, contextlib, io
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "benchmark"))
from cirq_defects import PROGRAMS
from lrq.llm import complete, usage_report
import cirq, numpy as np

MODELS = ["gpt-4o-mini", "gpt-4o", "claude-haiku-4-5", "claude-sonnet-4-6"]
TOL = 0.08


def runprog(code):
    ns = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(code, ns)
    return ns.get("RESULT")


def match(a, b):
    if a is None or b is None or type(a) != type(b):
        return False
    if isinstance(a, dict):
        ka = sum(a.values()) or 1; kb = sum(b.values()) or 1; keys = set(a) | set(b)
        return 0.5 * sum(abs(a.get(k, 0)/ka - b.get(k, 0)/kb) for k in keys) <= TOL
    if isinstance(a, (int, float)):
        return abs(a - b) < 1e-6
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(abs(x-y) < 1e-6 for x, y in zip(a, b))
    return a == b


def correct(code, ref):
    try:
        return match(runprog(code), ref)
    except Exception:
        return False


def extract(txt):
    m = re.findall(r"```(?:python)?\s*\n(.*?)```", txt, re.DOTALL)
    return (max(m, key=len) if m else txt).strip()


SYS = ("You are an expert Cirq engineer. Repair the program so it runs and produces "
       "the result its docstring specifies, preserving the author's intent. Keep the "
       "module-level variable RESULT. Output the COMPLETE corrected program in one "
       "```python block and nothing else.")
CIRQ_DOCS = ("- Build circuits from gates on cirq.LineQubit; cirq.Simulator().run(c, "
             "repetitions=n) samples.\n- result.measurements[key] is a (shots, nbits) array; "
             "bit i is qubit i.\n- Common gates: H, X, S, CNOT(ctrl,tgt), CZ, ry(theta) (radians).")


def repair(buggy, cfg, model):
    if cfg == "B0":
        p = f"The following Cirq program is buggy. Fix it.\n\n```python\n{buggy}\n```"
    elif cfg == "B0filler":   # length/retrieval-matched: docs as neutral background, no findings/cleanliness
        p = (f"The following Cirq program is buggy. Fix it.\n\nGeneral background (not specific "
             f"to this program):\n{CIRQ_DOCS}\n\n```python\n{buggy}\n```")
    else:  # B2 grounded: 'no structural defects' framing + retrieval + (implicit) validation
        p = (f"Repair this Cirq program.\n\nDetected issues:\n(static analysis reported no "
             f"structural defects)\n\nRelevant knowledge (retrieved):\n{CIRQ_DOCS}\n\n"
             f"```python\n{buggy}\n```")
    return extract(complete(SYS, p, temperature=0.0, max_tokens=1100, model=model))


def main(validate_only=False):
    refs = {}
    print("validating benchmark...")
    valid = []
    for pid, fam, corr, buggy in PROGRAMS:
        try:
            rc = runprog(corr); rb = runprog(buggy)
        except Exception as e:
            print(f"  ERR {pid}: {type(e).__name__}: {str(e)[:60]}"); continue
        refs[pid] = rc
        is_bug = not match(rb, rc)
        print(f"  {pid:18s} {fam:20s} buggy!=correct: {is_bug}")
        if is_bug:
            valid.append((pid, fam, corr, buggy))
    print(f"\n{len(valid)}/{len(PROGRAMS)} are genuine silent bugs.")
    if validate_only:
        return
    out = {}
    for model in MODELS:
        rec = {}
        def one(item):
            pid, fam, corr, buggy = item
            return pid, {"family": fam, **{c: correct(repair(buggy, c, model), refs[pid])
                                           for c in ["B0", "B2", "B0filler"]}}
        with ThreadPoolExecutor(max_workers=4) as ex:
            for pid, r in ex.map(one, valid):
                rec[pid] = r
        agg = {c: sum(1 for r in rec.values() if r[c]) for c in ["B0", "B2", "B0filler"]}
        out[model] = {**agg, "n": len(valid), "per_prog": rec}
        print(f"  {model:22s} B0={agg['B0']}/{len(valid)} B2={agg['B2']}/{len(valid)} "
              f"B0filler={agg['B0filler']}/{len(valid)}  B2-B0={agg['B2']-agg['B0']:+d}")
    print("\nusage:", usage_report())
    json.dump(out, open(os.path.join(HERE, "results", "cirqdefects.json"), "w"), indent=2)


if __name__ == "__main__":
    main(validate_only="--validate" in sys.argv)
