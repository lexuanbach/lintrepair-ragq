"""Independent real-bug replication: does the capability-governed crossover hold
on Bugs4Q (FSE-cited real Qiskit bugs we did NOT author)?

We use the 12 Bugs4Q bugs (7 crash + 5 silent ``output-wrong'') that reproduce
deterministically under their native Qiskit (0.45, a separate venv). Repair is
run with our B0/B2/B4 configs, but with a VERSION-NEUTRAL, minimal-change prompt
(the models know Qiskit 2.x; the bugs live in 0.x, so we neither steer toward nor
penalise modernisation). The oracle runs each candidate under the 0.45 interpreter
and requires it to run AND reproduce the human-fixed program's stdout.

Writes results/bugs4q_replication.json.
"""
from __future__ import annotations
import json, os, sys, subprocess, tempfile
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from lrq.detectors import detect
from lrq.retrieval import TfidfRetriever
from lrq.repair import _findings_block, extract_code
from lrq.llm import complete, usage_report

# Prefer the copy bundled in the artifact; fall back to /tmp for legacy runs.
_BUGS_PATH = os.path.join(HERE, "bugs4q_real.json")
if not os.path.exists(_BUGS_PATH):
    _BUGS_PATH = "/tmp/bugs4q_real.json"
BUGS = json.load(open(_BUGS_PATH))
# The Bugs4Q bugs execute under their native Qiskit 0.45; point this at a legacy
# interpreter (override with BUGS4Q_LEGACY_PY). Not needed to reproduce the frozen
# results in results/bugs4q_summary.json.
LEGACY_PY = os.environ.get("BUGS4Q_LEGACY_PY", "/tmp/bugs4q-venv/bin/python")
RETR = TfidfRetriever()
OUT = os.path.join(HERE, "results", "bugs4q_replication.json")
MODELS = ["gpt-4o-mini", "gpt-4o", "claude-haiku-4-5", "claude-sonnet-4-6"]

SYS = ("You are an expert Qiskit engineer. Repair the program so it runs correctly "
       "and preserves the author's evident intent. Make the MINIMAL change and keep "
       "the SAME Qiskit API style/version as the original program (do not modernise). "
       "Output the COMPLETE corrected program in one ```python block and nothing else.")


def run_legacy(code: str):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(code); p = fh.name
    try:
        r = subprocess.run([LEGACY_PY, p], capture_output=True, text=True, timeout=60)
        os.unlink(p)
        err = (r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "")
        return r.returncode, r.stdout, err
    except subprocess.TimeoutExpired:
        os.unlink(p); return -1, "", "timeout"


def correct(code, ref):
    rc, out, _ = run_legacy(code)
    return rc == 0 and out.strip() == ref.strip()


def repair(bug, cfg, model):
    src = bug["buggy"]
    findings = detect(src)
    docs = RETR.retrieve(src + " " + " ".join(f.category for f in findings), k=4)
    ctx = "\n".join(f"- [{d.tag}] {d.text}" for d in docs)
    if cfg == "B0":
        prompt = f"The following Qiskit program is buggy. Fix it.\n\n```python\n{src}\n```"
        return extract_code(complete(SYS, prompt, temperature=0.0, max_tokens=1200, model=model))
    debias = False if cfg == "B2" else "generic"
    history = ""; code = src
    for it in range(3):
        block = _findings_block(findings, debias)
        prompt = (f"Repair this Qiskit program.\n\nDetected issues:\n{block}\n\n"
                  f"Relevant knowledge:\n{ctx}\n{history}\n```python\n{src}\n```\n\n"
                  "Return the complete corrected program in one ```python block.")
        code = extract_code(complete(SYS, prompt, temperature=0.0, max_tokens=1300,
                                     sample=it, model=model))
        rc, _, err = run_legacy(code)
        if rc == 0:
            break
        history = f"\nYour previous attempt failed to run:\n{err}\nFix it.\n"
    return code


def run_model(model):
    rows = []
    def one(bug):
        rec = {"id": bug["id"], "kind": bug["kind"]}
        for cfg in ["B0", "B2", "B4"]:
            rec[cfg] = correct(repair(bug, cfg, model), bug["ref_stdout"])
        return rec
    with ThreadPoolExecutor(max_workers=4) as ex:
        rows = list(ex.map(one, BUGS))
    return rows


def main():
    out = {}
    n = len(BUGS)
    for model in MODELS:
        print(f"\n=== {model} on {n} Bugs4Q real bugs ===", flush=True)
        rows = run_model(model)
        out[model] = rows
        for cfg in ["B0", "B2", "B4"]:
            k = sum(1 for r in rows if r[cfg])
            kc = sum(1 for r in rows if r["kind"] == "crash" and r[cfg])
            ks = sum(1 for r in rows if r["kind"] == "output" and r[cfg])
            print(f"  {cfg}: {k}/{n}  (crash {kc}/7, silent {ks}/5)")
        print(f"  B2-B0 = {sum(r['B2'] for r in rows)-sum(r['B0'] for r in rows):+d}  usage={usage_report()}", flush=True)
        json.dump(out, open(OUT, "w"), indent=2)
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
