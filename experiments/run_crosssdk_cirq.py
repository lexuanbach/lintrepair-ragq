"""Cross-SDK probe (external validity): does the grounding-context interference
on strong models reproduce on CIRQ (a non-Qiskit SDK)?

We could not source this from QuanBench+ (its released Cirq/PennyLane canonical
solutions are all written in Qiskit -- a dataset defect), so we author a small,
execution-verified Cirq silent-bug set: 8 programs with a wrong-gate/angle/axis
bug that runs but yields a wrong distribution. Each program defines `circuit`
(a cirq.Circuit measuring all qubits into key 'result'); the oracle is TVD<=0.08
against the corrected version's distribution (same oracle as Q-Defects40).

We run B0 (plain repair) vs B2 (grounded: a ``no structural defects'' framing +
neutral context -- the SDK-agnostic core of our grounding) across a weak, mid, and
strong model, and check the crossover sign. Writes results/crosssdk_cirq.json.
"""
from __future__ import annotations
import json, os, sys, re
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT)
from lrq.llm import complete, usage_report
import cirq, numpy as np

SHOTS = 4000; TOL = 0.08
MODELS = ["gpt-4o-mini", "gpt-4o", "claude-sonnet-4-6"]

# Each: correct vs buggy. Bug is SILENT (runs, wrong distribution). `circuit` is a cirq.Circuit.
def _b(intent, code):
    return f"# Goal: {intent}\n{code}"
BUGS = [
 ("bell_cz_for_cx",
  "q=cirq.LineQubit.range(2)\ncircuit=cirq.Circuit([cirq.H(q[0]),cirq.CNOT(q[0],q[1]),cirq.measure(*q,key='result')])",
  _b("prepare a Bell state (entangled; only outcomes 00 and 11, equal).",
     "q=cirq.LineQubit.range(2)\ncircuit=cirq.Circuit([cirq.H(q[0]),cirq.CZ(q[0],q[1]),cirq.measure(*q,key='result')])")),
 ("ghz_missing_cnot",
  "q=cirq.LineQubit.range(3)\ncircuit=cirq.Circuit([cirq.H(q[0]),cirq.CNOT(q[0],q[1]),cirq.CNOT(q[1],q[2]),cirq.measure(*q,key='result')])",
  _b("prepare a 3-qubit GHZ state (only outcomes 000 and 111, equal).",
     "q=cirq.LineQubit.range(3)\ncircuit=cirq.Circuit([cirq.H(q[0]),cirq.CNOT(q[0],q[1]),cirq.measure(*q,key='result')])")),
 ("rot_wrong_angle",
  "import numpy as np\nq=cirq.LineQubit.range(1)\ncircuit=cirq.Circuit([cirq.ry(np.pi/2)(q[0]),cirq.measure(*q,key='result')])",
  _b("put qubit 0 into an equal superposition: P(0)=P(1)=0.5.",
     "import numpy as np\nq=cirq.LineQubit.range(1)\ncircuit=cirq.Circuit([cirq.ry(np.pi/4)(q[0]),cirq.measure(*q,key='result')])")),
 ("swapped_ctrl_target",
  "q=cirq.LineQubit.range(2)\ncircuit=cirq.Circuit([cirq.X(q[0]),cirq.CNOT(q[0],q[1]),cirq.measure(*q,key='result')])",
  _b("set qubit 0 to |1>, then copy it to qubit 1 via CNOT (control=q0). Expect 11.",
     "q=cirq.LineQubit.range(2)\ncircuit=cirq.Circuit([cirq.X(q[0]),cirq.CNOT(q[1],q[0]),cirq.measure(*q,key='result')])")),
 ("wrong_axis_rx_ry",
  "import numpy as np\nq=cirq.LineQubit.range(1)\ncircuit=cirq.Circuit([cirq.H(q[0]),cirq.rz(np.pi/2)(q[0]),cirq.H(q[0]),cirq.measure(*q,key='result')])",
  _b("apply H, then a Z-axis phase rotation Rz(pi/2), then H on qubit 0.",
     "import numpy as np\nq=cirq.LineQubit.range(1)\ncircuit=cirq.Circuit([cirq.H(q[0]),cirq.rx(np.pi/2)(q[0]),cirq.H(q[0]),cirq.measure(*q,key='result')])")),
 ("phase_missing_s",
  "q=cirq.LineQubit.range(1)\ncircuit=cirq.Circuit([cirq.H(q[0]),cirq.S(q[0]),cirq.H(q[0]),cirq.measure(*q,key='result')])",
  _b("apply H, then the S gate, then H on qubit 0.",
     "q=cirq.LineQubit.range(1)\ncircuit=cirq.Circuit([cirq.H(q[0]),cirq.H(q[0]),cirq.measure(*q,key='result')])")),
 ("w3_wrong_first_rot",
  "import numpy as np\nq=cirq.LineQubit.range(2)\ncircuit=cirq.Circuit([cirq.ry(2*np.arccos(np.sqrt(2/3)))(q[0]),cirq.measure(*q,key='result')])",
  _b("rotate qubit 0 with Ry(2*arccos(sqrt(2/3))) so that P(1)=1/3.",
     "import numpy as np\nq=cirq.LineQubit.range(2)\ncircuit=cirq.Circuit([cirq.ry(np.pi/2)(q[0]),cirq.measure(*q,key='result')])")),
 ("cnot_chain_wrong_target",
  "q=cirq.LineQubit.range(3)\ncircuit=cirq.Circuit([cirq.H(q[0]),cirq.CNOT(q[0],q[1]),cirq.X(q[2]),cirq.measure(*q,key='result')])",
  _b("H on q0, CNOT(q0->q1), and X on q2. Expect 001 and 111 (q2 always 1).",
     "q=cirq.LineQubit.range(3)\ncircuit=cirq.Circuit([cirq.H(q[0]),cirq.CNOT(q[0],q[2]),cirq.X(q[2]),cirq.measure(*q,key='result')])")),
]


def run_cirq(code: str):
    import contextlib, io
    ns = {"cirq": cirq, "np": np}
    with contextlib.redirect_stdout(io.StringIO()):   # ignore any prints in generated code
        exec(code, ns)
    circ = ns["circuit"]
    res = cirq.Simulator(seed=1).run(circ, repetitions=SHOTS)
    rows = res.measurements["result"]
    out = {}
    for r in rows:
        k = "".join(str(int(b)) for b in r); out[k] = out.get(k, 0) + 1
    return out


def tvd(a, b):
    ka = sum(a.values()) or 1; kb = sum(b.values()) or 1; keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0) / ka - b.get(k, 0) / kb) for k in keys)


def extract(txt):
    m = re.findall(r"```(?:python)?\s*\n(.*?)```", txt, re.DOTALL)
    return (max(m, key=len) if m else txt).strip()


SYS = ("You are an expert Cirq engineer. Repair the program so it runs and implements "
       "the author's intended computation. Keep the variable `circuit` (a cirq.Circuit "
       "measuring all qubits into key 'result'). Output the COMPLETE corrected program "
       "in one ```python block and nothing else.")


def repair(buggy, cfg, model):
    if cfg == "B0":
        prompt = f"The following Cirq program is buggy. Fix it.\n\n```python\n{buggy}\n```"
    else:  # B2: grounded with a (truthful for silent bugs) 'no structural defects' framing + context
        prompt = (f"Repair this Cirq program.\n\nDetected issues:\n(static analysis reported no "
                  f"structural defects)\n\nRelevant knowledge (retrieved):\n- Cirq circuits are "
                  f"built from gates on LineQubits; measure with cirq.measure(*qubits, key='result').\n"
                  f"- The simulator samples bitstrings from the final state.\n\n```python\n{buggy}\n```")
    return extract(complete(SYS, prompt, temperature=0.0, max_tokens=900, model=model))


def correct(code, ref):
    try:
        return tvd(run_cirq(code), ref) <= TOL
    except Exception:
        return False


def main():
    refs = {name: run_cirq(corr) for name, corr, _ in BUGS}
    # keep only genuinely-silent bugs (buggy runs but FAILS the oracle)
    valid = [(n, c, b) for n, c, b in BUGS if not correct(b, refs[n])]
    print(f"{len(valid)}/{len(BUGS)} Cirq programs are genuine silent bugs. shots={SHOTS}\n")
    out = {}
    for model in MODELS:
        rec = {}
        def one(item):
            name, corr, buggy = item
            return name, {c: correct(repair(buggy, c, model), refs[name]) for c in ["B0", "B2"]}
        with ThreadPoolExecutor(max_workers=4) as ex:
            for name, r in ex.map(one, valid):
                rec[name] = r
        b0 = sum(r["B0"] for r in rec.values()); b2 = sum(r["B2"] for r in rec.values())
        out[model] = {"B0": b0, "B2": b2, "n": len(valid), "per_bug": rec}
        print(f"  {model:22s} B0={b0}/{len(valid)}  B2={b2}/{len(valid)}  B2-B0={b2-b0:+d}")
    print("\nusage:", usage_report())
    json.dump(out, open(os.path.join(HERE, "results", "crosssdk_cirq.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
