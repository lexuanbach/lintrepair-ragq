"""Score LintQ's SARIF output on Q-Defects40 against ground-truth labels.

Program-level detection (matches the paper's Table I convention): a program is
"flagged" if LintQ emits >=1 warning on it. Precision/recall/F1 over the 32
defective + 8 clean programs, plus per-family recall and the exact rules fired.
Usage: python score_lintq.py results.sarif
"""
import json, sys, os
from collections import defaultdict, Counter

SARIF = sys.argv[1] if len(sys.argv) > 1 else "/tmp/qd40/results.sarif"
# Ground-truth labels are bundled next to this script; /tmp is a legacy fallback.
_HERE = os.path.dirname(os.path.abspath(__file__))
_LABELS = os.path.join(_HERE, "labels.json")
if not os.path.exists(_LABELS):
    _LABELS = "/tmp/qd40/labels.json"
labels = json.load(open(_LABELS))

sar = json.load(open(SARIF))
fired = defaultdict(set)        # program id -> {ruleId,...}
for run in sar.get("runs", []):
    for r in run.get("results", []):
        rule = r.get("ruleId") or r.get("rule", {}).get("id", "?")
        for loc in r.get("locations", []):
            uri = loc.get("physicalLocation", {}).get("artifactLocation", {}).get("uri", "")
            pid = os.path.splitext(os.path.basename(uri))[0]
            if pid:
                fired[pid].add(rule)

ids = sorted(labels)
defective = [i for i in ids if labels[i]["kind"] != "clean"]
clean = [i for i in ids if labels[i]["kind"] == "clean"]

flagged = {i for i in ids if fired.get(i)}
TP = len(flagged & set(defective))
FP = len(flagged & set(clean))
FN = len(set(defective) - flagged)
TN = len(set(clean) - flagged)
prec = TP / (TP + FP) if (TP + FP) else float("nan")
rec = TP / (TP + FN) if (TP + FN) else float("nan")
f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else float("nan")

print(f"LintQ on Q-Defects40 (program-level, high-precision suite)")
print(f"  defective={len(defective)} clean={len(clean)}")
print(f"  TP={TP} FP={FP} FN={FN} TN={TN}")
print(f"  precision={prec:.3f} recall={rec:.3f} F1={f1:.3f}")
print()
print("Per-family recall (flagged / total):")
fam = defaultdict(lambda: [0, 0])
for i in defective:
    cat = labels[i]["category"]
    fam[cat][1] += 1
    if i in flagged:
        fam[cat][0] += 1
for cat in sorted(fam):
    k, n = fam[cat]
    print(f"  {cat:28s} {k}/{n}")
print()
print("Rules fired (rule -> #programs):")
rc = Counter(r for s in fired.values() for r in s)
for r, n in rc.most_common():
    print(f"  {n:2d}  {r}")
print()
print("Per-program detail:")
for i in ids:
    tag = "CLEAN" if labels[i]["kind"] == "clean" else labels[i]["category"]
    hit = "FLAG" if i in flagged else "    "
    rules = ",".join(sorted(s.split("/")[-1] for s in fired.get(i, []))) or "-"
    print(f"  [{hit}] {i:34s} {tag:28s} {rules}")
