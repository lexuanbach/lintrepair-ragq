"""Pass@k + anchoring-under-sampling (reviewer Q4). On Sonnet (the most anchored
model) over the 32 defective programs, compute pass@1 (greedy) and pass@5
(temperature 0.6, 5 samples) for B0 (unguided) and B2 (naive grounding). If the
anchoring (B2<B0) persists under pass@5, it is not a greedy-decoding artifact.
Writes results/passk.json."""
import json, os, sys
from concurrent.futures import ThreadPoolExecutor
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE); sys.path.insert(0,ROOT)
from lrq import repair as R
from lrq.detectors import detect, Finding
from lrq.harness import run_script, results_match
from lrq.llm import usage_report
BENCH=[b for b in json.load(open(os.path.join(ROOT,"benchmark","benchmark.json"))) if b["kind"]!="clean"]
FROZEN={r["id"]:r for r in json.load(open(os.path.join(HERE,"results","records_FROZEN.json")))}
MODEL="claude-sonnet-4-6"; K=5
def findings(inst):
    st=detect(inst["buggy"]); cats={f.category for f in st}; out=list(st)
    for c in FROZEN[inst["id"]]["detection"]["union"]:
        if c not in cats: out.append(Finding(c,0,"(LLM)"))
    return out
def ok(code,ref):
    o=run_script(code); return bool(results_match(ref,o.result)[0]) if o.status=="ok" else False
def one(inst):
    ref=inst["reference_result"]; f=findings(inst)
    b0=[ok(R.plain_repair(inst["buggy"],sample=s,model=MODEL).code,ref) for s in range(K)]
    b2=[ok(R.full_repair(inst["buggy"],f,run_script,sample=s,max_iters=3,debias=False,model=MODEL).code,ref) for s in range(K)]
    return {"id":inst["id"],"kind":inst["kind"],
            "B0_pass1":b0[0],"B0_pass5":any(b0),"B2_pass1":b2[0],"B2_pass5":any(b2)}
with ThreadPoolExecutor(max_workers=4) as ex:
    rows=list(ex.map(one,BENCH))
n=len(rows)
for tag in ["B0_pass1","B0_pass5","B2_pass1","B2_pass5"]:
    print(f"  Sonnet {tag}: {sum(1 for r in rows if r[tag])}/{n}")
print("  usage:",usage_report())
json.dump(rows,open(os.path.join(HERE,"results","passk.json"),"w"),indent=2)
