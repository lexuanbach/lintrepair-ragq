"""Approach #2: mutation injection. Generate silent-semantic bugs by mutating the
HUMAN-FIXED (correct) Q-Defects40 programs -- realistic multi-line programs, so the
injected bug is buried (the difficulty-control lesson). Each mutant is validated by
execution against the program's reference output: a mutant that RUNS but diverges
is a genuine SILENT bug; one that crashes is a crash bug; one that matches is an
equivalent mutant (discarded). Writes results/mutation_dataset.json.
"""
from __future__ import annotations
import json, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT)
from lrq.harness import run_script, results_match
BENCH = [b for b in json.load(open(os.path.join(ROOT, "benchmark", "benchmark.json"))) if b["kind"] != "clean"]

GATE_SUBS = [("cx", "cz"), ("cz", "cx"), ("cx", "cy"), ("h", "x"), ("x", "y"), ("s", "t"), ("cx", "ch")]


def mutants(src):
    """Yield (op, mutated_src) for silent-leaning mutation operators."""
    # 1. gate substitution: qc.G( -> qc.H(   (one occurrence at a time)
    for g, h in GATE_SUBS:
        for m in re.finditer(rf"\.{g}\(", src):
            yield f"gate:{g}->{h}", src[:m.start()] + f".{h}(" + src[m.end():]
    # 2. swap the two qubit args of a 2-qubit gate:  .cx(a, b) -> .cx(b, a)
    for m in re.finditer(r"\.(cx|cz|cy|ch|swap)\(\s*([^,()]+?)\s*,\s*([^,()]+?)\s*\)", src):
        g, a, b = m.group(1), m.group(2), m.group(3)
        yield f"swap:{g}", src[:m.start()] + f".{g}({b}, {a})" + src[m.end():]
    # 3. scale a rotation angle by 0.5:  .rx(theta -> .rx(0.5*(theta)
    for m in re.finditer(r"\.(rx|ry|rz|p|u)\(\s*([^,()]+?)\s*([,)])", src):
        g, ang, tail = m.group(1), m.group(2), m.group(3)
        yield f"angle:{g}", src[:m.start()] + f".{g}(0.5*({ang}){tail}" + src[m.end():]


def main():
    out = []; silent = crash = equiv = total = 0
    for inst in BENCH:
        fixed = inst["fixed"]; ref = inst["reference_result"]
        seen = set()
        for op, mut in mutants(fixed):
            if mut in seen:
                continue
            seen.add(mut); total += 1
            o = run_script(mut)
            if o.status != "ok":
                crash += 1; continue
            ok = results_match(ref, o.result)[0]
            if ok:
                equiv += 1
            else:
                silent += 1
                out.append({"base_id": inst["id"], "op": op, "kind": "silent",
                            "buggy": mut, "reference": ref})
    json.dump({"summary": {"base_programs": len(BENCH), "mutants_tried": total,
                           "silent_bugs": silent, "crash_bugs": crash, "equivalent": equiv},
               "silent_bugs": out}, open(os.path.join(HERE, "results", "mutation_dataset.json"), "w"), indent=2)
    print(f"Mutation injection over {len(BENCH)} correct programs:")
    print(f"  mutants tried:   {total}")
    print(f"  SILENT bugs:     {silent}  <- usable silent-semantic dataset")
    print(f"  crash bugs:      {crash}")
    print(f"  equivalent (drop): {equiv}")
    from collections import Counter
    print("  silent bugs by operator:", dict(Counter(r["op"].split(":")[0] for r in out)))


if __name__ == "__main__":
    main()
