"""Deterministic scripted-migration baseline (reviewer Q96 / R2): a non-LLM tool
that applies the well-known Qiskit 1.0 API migrations (execute->backend.run,
qiskit.Aer->qiskit_aer, bind_parameters->assign_parameters). It shows what a
scripted migration can and cannot fix: the API-removal crash families vs the
logic/semantic families. Run through the same execution oracle. No API.
Writes results/scripted_baseline.json.
"""
from __future__ import annotations
import json, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT)
from lrq.harness import run_script, results_match
BENCH = [b for b in json.load(open(os.path.join(ROOT, "benchmark", "benchmark.json"))) if b["kind"] != "clean"]


def _balanced(s, open_idx):
    """Return index just past the ) matching the ( at open_idx."""
    depth = 0
    for j in range(open_idx, len(s)):
        if s[j] == "(":
            depth += 1
        elif s[j] == ")":
            depth -= 1
            if depth == 0:
                return j + 1
    return len(s)


def _split_top(arglist):
    """Split a comma-separated arg list at top-level commas only."""
    out, depth, cur = [], 0, ""
    for ch in arglist:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur.strip()); cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def migrate(src: str) -> str:
    s = src
    s = s.replace(".bind_parameters(", ".assign_parameters(")          # deprecated-bind
    s = s.replace("qiskit.execute(", "execute(").replace("qiskit.Aer", "Aer")   # de-qualify
    # imports: drop removed names, add qiskit_aer + transpile
    def fix_imp(m):
        names = [n.strip() for n in m.group(1).split(",")]
        keep = [n for n in names if n not in {"execute", "Aer", "assemble"}]
        if "transpile" not in keep:
            keep.append("transpile")
        out = [f"from qiskit import {', '.join(keep)}"] if keep else []
        out.append("from qiskit_aer import AerSimulator")
        return "\n".join(out)
    s = re.sub(r"from qiskit import ([^\n]+)", fix_imp, s)            # all import lines
    # dedupe repeated import lines (keep first occurrence)
    seen, lines = set(), []
    for ln in s.splitlines():
        key = ln.strip()
        if key.startswith(("from qiskit import", "from qiskit_aer import AerSimulator")) and key in seen:
            continue
        seen.add(key); lines.append(ln)
    s = "\n".join(lines)
    if "from qiskit_aer import AerSimulator" not in s:
        s = "from qiskit_aer import AerSimulator\n" + s
    s = re.sub(r"Aer\.get_backend\([^)]*\)", "AerSimulator()", s)       # backend factory
    # execute(...) with balanced parens -> a sim var + sim.run(transpile(...))
    counter = [0]
    while "execute(" in s:
        i = s.index("execute(")
        end = _balanced(s, i + len("execute") )
        inner = s[i + len("execute(") : end - 1]
        args = _split_top(inner)
        qc = args[0]; backend = args[1]
        shots = next((a for a in args if a.startswith("shots")), "shots=1024")
        seed = next((a for a in args if "seed" in a), None)
        if "AerSimulator()" in backend or "(" in backend:             # inline backend -> temp var
            counter[0] += 1; var = f"_sim{counter[0]}"
            ctor = f"AerSimulator({seed})" if seed else "AerSimulator()"
            prefix = f"{var} = {ctor}\n"
            backend = var
        else:
            prefix = ""
            if seed:                                                   # push seed into the named backend ctor
                s = s.replace(f"{backend} = AerSimulator()", f"{backend} = AerSimulator({seed})", 1)
        repl = f"{backend}.run(transpile({qc}, {backend}), {shots})"
        s = s[:i] + repl + s[end:]
        if prefix:                                                     # insert temp-sim line above the statement
            line_start = s.rfind("\n", 0, i) + 1
            s = s[:line_start] + prefix + s[line_start:]
    return s


def main():
    out = []; fam = {}
    for inst in BENCH:
        cand = migrate(inst["buggy"])
        o = run_script(cand)
        good = bool(results_match(inst["reference_result"], o.result)[0]) if o.status == "ok" else False
        out.append({"id": inst["id"], "category": inst["category"], "kind": inst["kind"], "fixed": good})
        fam.setdefault(inst["category"], [0, 0]); fam[inst["category"]][1] += 1; fam[inst["category"]][0] += int(good)
    n = len(out); k = sum(r["fixed"] for r in out)
    print(f"Scripted-migration baseline: {k}/{n} defects fixed (no LLM)")
    print("by family:")
    for c, (a, b) in sorted(fam.items()):
        print(f"  {c:28s} {a}/{b}")
    json.dump({"total": f"{k}/{n}", "by_family": {c: f"{a}/{b}" for c, (a, b) in fam.items()}, "records": out},
              open(os.path.join(HERE, "results", "scripted_baseline.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
