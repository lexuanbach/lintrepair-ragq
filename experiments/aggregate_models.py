"""Unify frozen (Sonnet/Haiku) + extra (gpt-4o-mini, gemini, qwen, llama) repair
results into one cross-model table and test the two-factor model across the
full capability spectrum (validator runnability gain vs base reliability rho),
plus a pooled, higher-powered B3-vs-B2 (debiasing > naive grounding) test.

Writes results/cross_model.json. No API calls.
"""
from __future__ import annotations
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from lrq.stats import wilson, paired_bootstrap_diff, mcnemar

R = os.path.join(HERE, "results")
frozen = {r["id"]: r for r in json.load(open(os.path.join(R, "records_FROZEN.json"))) if r["kind"] != "clean"}
extra = json.load(open(os.path.join(R, "models_extra.json")))
# B4 supplement for the two Anthropic models (B4 postdates the frozen pilot).
try:
    b4sup = json.load(open(os.path.join(R, "b4_anthropic.json")))
except FileNotFoundError:
    b4sup = {}
ids = sorted(frozen.keys())

# per-model per-instance sem/ran vectors over the 32 defective programs (aligned by id)
def frozen_vec(model, cfg, field):
    if cfg == "B4":                       # from the supplement, not the frozen file
        return [1 if b4sup[model][i][field] else 0 for i in ids]
    return [1 if frozen[i]["repair"][model][cfg]["pass1"][field] else 0 for i in ids]

def extra_vec(model, cfg, field):
    rec = {r["id"]: r for r in extra[model]["records"]}
    return [1 if rec[i][cfg][field] else 0 for i in ids]

SONNET = "claude-sonnet-4-6"; HAIKU = "claude-haiku-4-5-20251001"
MODELS = [
    ("Sonnet 4.6", SONNET, "frozen", ["B0", "B1", "B2", "B3", "B4"]),
    ("Haiku 4.5", HAIKU, "frozen", ["B0", "B2", "B3", "B4"]),
    ("GPT-4o", "gpt-4o", "extra", ["B0", "B1", "B2", "B3", "B4"]),
    ("Gemini 2.5 Pro", "gemini-2.5-pro", "extra", ["B0", "B1", "B2", "B3", "B4"]),
    ("Gemini 2.5 Flash", "gemini-2.5-flash", "extra", ["B0", "B1", "B2", "B3", "B4"]),
    ("Gemini 2.5 Flash-Lite", "gemini-2.5-flash-lite", "extra", ["B0", "B1", "B2", "B3", "B4"]),
    ("GPT-4o-mini", "gpt-4o-mini", "extra", ["B0", "B1", "B2", "B3", "B4"]),
    ("Gemma2-9B", "gemma2:9b", "extra", ["B0", "B2", "B3", "B4"]),
    ("Qwen2.5-7B", "qwen2.5:7b", "extra", ["B0", "B2", "B3", "B4"]),
    ("Llama-3.1-8B", "llama3.1:8b", "extra", ["B0", "B2", "B3", "B4"]),
    ("Mistral-7B", "mistral:7b", "extra", ["B0", "B2", "B3", "B4"]),
    ("Llama-3.2-3B", "llama3.2:3b", "extra", ["B0", "B2", "B3", "B4"]),
    ("Qwen2.5-Coder-7B", "qwen2.5-coder:7b", "extra", ["B0", "B2", "B3", "B4"]),
    ("Qwen2.5-14B", "qwen2.5:14b", "extra", ["B0", "B2", "B3", "B4"]),
    ("Command-R-35B", "command-r:35b", "extra", ["B0", "B2", "B3", "B4"]),
    ("Qwen2.5-Coder-32B", "qwen2.5-coder:32b", "extra", ["B0", "B2", "B3", "B4"]),
    ("Llama-3.3-70B", "llama3.3:70b", "extra", ["B0", "B2", "B3", "B4"]),
    ("Qwen2.5-72B", "qwen2.5:72b", "extra", ["B0", "B2", "B3", "B4"]),
]

def vec(model, src, cfg, field):
    return frozen_vec(model, cfg, field) if src == "frozen" else extra_vec(model, cfg, field)

rows = []
pooled = {"B3_vs_B2": ([], []), "B2_vs_B0": ([], []), "B4_vs_B2": ([], []), "B3_vs_B4": ([], [])}
for disp, key, src, cfgs in MODELS:
    if src == "extra" and key not in extra:
        print(f"  (skip {disp}: not run yet)"); continue
    row = {"model": disp, "id": key, "configs": cfgs}
    for cfg in cfgs:
        s = vec(key, src, cfg, "sem"); n = len(s)
        p, lo, hi = wilson(sum(s), n)
        e = sum(vec(key, src, cfg, "ran")) / n
        row[cfg] = {"k": sum(s), "n": n, "p": round(p, 3), "ci": [round(lo, 3), round(hi, 3)], "exec": round(e, 3)}
    row["rho_hat"] = row["B0"]["exec"]
    if "B2" in cfgs:
        row["valid_exec_gain"] = round(row["B2"]["exec"] - row["B0"]["exec"], 3)
    PAIRS = [("B3", "B2"), ("B2", "B0"), ("B4", "B2"), ("B3", "B4")]
    for a, b in PAIRS:
        if a in cfgs and b in cfgs:
            va = vec(key, src, a, "sem"); vb = vec(key, src, b, "sem")
            d, lo, hi = paired_bootstrap_diff(va, vb); mc = mcnemar(vb, va)
            row[f"{a}_vs_{b}"] = {"diff": round(d, 3), "ci": [round(lo, 3), round(hi, 3)],
                                  "p": round(mc[2], 4), "wl": [mc[0], mc[1]]}
            pooled[f"{a}_vs_{b}"][0].extend(va); pooled[f"{a}_vs_{b}"][1].extend(vb)
    rows.append(row)

def pool(a, b):
    d, lo, hi = paired_bootstrap_diff(a, b); mc = mcnemar(b, a)
    return {"n_pairs": len(a), "diff": round(d, 3), "ci": [round(lo, 3), round(hi, 3)],
            "p": round(mc[2], 4), "wins": mc[0], "losses": mc[1]}

summary = {"rows": rows, "pooled": {k: pool(*v) for k, v in pooled.items()}}
json.dump(summary, open(os.path.join(R, "cross_model.json"), "w"), indent=2)

# console: capability-governed crossover, sorted by rho
print(f"\n{'model':22s} {'rho':>5} {'B0':>5} {'B2':>5} {'B3':>5} {'B4':>5} {'B2-B0':>6} {'B4-B2':>6}")
for r in sorted(rows, key=lambda x: x["rho_hat"]):
    g = lambda c: r.get(c, {}).get("p", float("nan"))
    b2b0 = r.get("B2_vs_B0", {}).get("diff", "")
    b4b2 = r.get("B4_vs_B2", {}).get("diff", "")
    print(f"{r['model']:22s} {r['rho_hat']:5.2f} {g('B0'):5.3f} {g('B2'):5.3f} {g('B3'):5.3f} {g('B4'):5.3f} {str(b2b0):>6} {str(b4b2):>6}")
print("\nPOOLED (treatment vs baseline; diff, CI, McNemar p, wins/losses):")
for k, v in summary["pooled"].items():
    print(f"  {k:10s} n={v['n_pairs']:3d} diff={v['diff']:+.3f} CI{v['ci']} p={v['p']:.4f} w/l={v['wins']}/{v['losses']}")
print("\nwrote", os.path.join(R, "cross_model.json"))
