"""Revision analyses (no API calls; reads the frozen records):
  1. capability x grounding INTERACTION as a model-level regression of the
     grounding gain (B2-B0 pass@1) on base capability rho-hat, with a bootstrap CI;
  2. breakdown of the pooled McNemar discordant pairs (wins/losses) by defect
     kind (crash vs silent-semantic) and by capability tier.

Writes results/interaction_and_losses.json. Reproduces the pooled 168/76.
"""
from __future__ import annotations
import json, os, math

HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "results")
frozen = {r["id"]: r for r in json.load(open(os.path.join(R, "records_FROZEN.json")))
          if r["kind"] != "clean"}
extra = json.load(open(os.path.join(R, "models_extra.json")))
ids = sorted(frozen.keys())
kind = {i: frozen[i]["kind"] for i in ids}           # "crash" | "semantic"
SON, HAI = "claude-sonnet-4-6", "claude-haiku-4-5-20251001"

def fvec(m, cfg, f): return [1 if frozen[i]["repair"][m][cfg]["pass1"][f] else 0 for i in ids]
def evec(m, cfg, f):
    rec = {r["id"]: r for r in extra[m]["records"]}
    return [1 if rec[i][cfg][f] else 0 for i in ids]

models = [("Sonnet 4.6", SON, "f"), ("Haiku 4.5", HAI, "f")] + [(m, m, "e") for m in extra]
def vec(src, m, cfg, f): return fvec(m, cfg, f) if src == "f" else evec(m, cfg, f)

rho, gain, wins, losses = [], [], 0, 0
loss_kind = {"crash": 0, "semantic": 0}; win_kind = {"crash": 0, "semantic": 0}
loss_kind_cap = {}
for name, m, src in models:
    b0, b2 = vec(src, m, "B0", "sem"), vec(src, m, "B2", "sem")
    r = sum(vec(src, m, "B0", "ran")) / 32
    rho.append(r); gain.append((sum(b2) - sum(b0)) / 32)
    for idx, (a, b) in enumerate(zip(b0, b2)):
        k = kind[ids[idx]]
        if b == 1 and a == 0: wins += 1; win_kind[k] += 1
        if a == 1 and b == 0:
            losses += 1; loss_kind[k] += 1
            cap = "high" if r >= 0.5 else "low"
            loss_kind_cap[f"{k}:{cap}"] = loss_kind_cap.get(f"{k}:{cap}", 0) + 1

# OLS gain ~ rho  (interaction slope)
n = len(rho); mx = sum(rho) / n; my = sum(gain) / n
sxx = sum((x - mx) ** 2 for x in rho); sxy = sum((x - mx) * (y - my) for x, y in zip(rho, gain))
slope = sxy / sxx; intercept = my - slope * mx
ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(rho, gain))
ss_tot = sum((y - my) ** 2 for y in gain); r2 = 1 - ss_res / ss_tot
# deterministic bootstrap CI (fixed LCG seed; no numpy dependency)
seed = 12345
def rnd():
    global seed; seed = (1103515245 * seed + 12345) & 0x7fffffff; return seed / 0x7fffffff
bs = []
for _ in range(20000):
    idx = [int(rnd() * n) for _ in range(n)]
    xs = [rho[i] for i in idx]; ys = [gain[i] for i in idx]
    m2 = sum(xs) / n; sx = sum((x - m2) ** 2 for x in xs)
    if sx == 0: continue
    bs.append(sum((x - m2) * (y - sum(ys) / n) for x, y in zip(xs, ys)) / sx)
bs.sort(); lo, hi = bs[int(.025 * len(bs))], bs[int(.975 * len(bs))]
cross = -intercept / slope

out = {
    "pooled": {"wins": wins, "losses": losses, "pairs": n * 32},
    "interaction": {"slope": round(slope, 3), "intercept": round(intercept, 3),
                    "r2": round(r2, 3), "boot95": [round(lo, 3), round(hi, 3)],
                    "gain_at_rho0": round(intercept, 3),
                    "gain_at_rho1": round(intercept + slope, 3),
                    "zero_crossing_rho": round(cross, 3)},
    "wins_by_kind": win_kind, "losses_by_kind": loss_kind,
    "losses_by_kind_capability": loss_kind_cap,
    "losses_high_cap": sum(v for k, v in loss_kind_cap.items() if k.endswith("high")),
}
# detector-hit audit: does every flagged program name its TRUE family?
allrecs = json.load(open(os.path.join(R, "records_FROZEN.json")))
defec = [r for r in allrecs if r["kind"] != "clean" and "detection" in r]
clean = [r for r in allrecs if r["kind"] == "clean" and "detection" in r]
audit = {}
for key in ("static", "llm", "union"):
    flagged = [r for r in defec if r["detection"][key]]
    right = sum(1 for r in flagged if r["category"] in r["detection"][key])
    audit[key] = {"flagged": len(flagged), "right_family": right,
                  "false_pos_on_clean": sum(1 for r in clean if r["detection"][key])}
out["detector_audit"] = audit

json.dump(out, open(os.path.join(R, "interaction_and_losses.json"), "w"), indent=2)
print(json.dumps(out, indent=2))
