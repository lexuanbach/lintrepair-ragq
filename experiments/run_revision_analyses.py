"""Consolidated revision analyses (ICSE major-revision response). Regenerates,
from the frozen + extra + B4 + detector-sensitivity records (no new API calls;
LLM-detector calls on negatives are cached), all numbers added in revision:

  - family_level   : per-family pooled pass@1 for B0/B2/B3/B4 (11 models)
  - strong_family  : Sonnet/Haiku silent-semantic family breakdown (anchoring cells)
  - stratified_kind: pooled B2-B0 split by crash vs semantic defects (paired test)
  - capability_corr: Pearson/Spearman of base capability rho-hat vs grounding gain
  - expanded_negatives: detection false positives on 8 clean + 32 matched human fixes

Writes results/revision_analyses.json.
"""
from __future__ import annotations
import json, os, sys, math
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from lrq.stats import paired_bootstrap_diff, mcnemar
R = os.path.join(HERE, "results")
frozen = {r["id"]: r for r in json.load(open(f"{R}/records_FROZEN.json")) if r["kind"] != "clean"}
extra = json.load(open(f"{R}/models_extra.json"))
b4 = json.load(open(f"{R}/b4_anthropic.json"))
ids = sorted(frozen); cat = {i: frozen[i]["category"] for i in ids}; kind = {i: frozen[i]["kind"] for i in ids}
SON = "claude-sonnet-4-6"; HAI = "claude-haiku-4-5-20251001"
MODELS = [(SON, "frozen", 1.0), (HAI, "frozen", 0.875), ("gpt-4o", "extra", 0.5),
          ("gpt-4o-mini", "extra", 0.0), ("gemini-2.5-pro", "extra", 0.406),
          ("gemini-2.5-flash", "extra", 0.781), ("gemini-2.5-flash-lite", "extra", 0.688),
          ("gemma2:9b", "extra", 0.469), ("qwen2.5:7b", "extra", 0.469),
          ("llama3.1:8b", "extra", 0.188), ("mistral:7b", "extra", 0.156),
          ("llama3.2:3b", "extra", 0.438), ("qwen2.5-coder:7b", "extra", 0.531),
          ("qwen2.5:14b", "extra", 0.125), ("command-r:35b", "extra", 0.469),
          ("qwen2.5-coder:32b", "extra", 0.562), ("llama3.3:70b", "extra", 0.594),
          ("qwen2.5:72b", "extra", 0.844)]


def sem(mk, src, cfg, i):
    if cfg == "B4" and src == "frozen":
        return 1 if b4[mk][i]["sem"] else 0
    if src == "frozen":
        return 1 if frozen[i]["repair"][mk][cfg]["pass1"]["sem"] else 0
    rec = {r["id"]: r for r in extra[mk]["records"]}
    return 1 if (cfg in rec[i] and rec[i][cfg]["sem"]) else 0


def pearson(a, b):
    n = len(a); ma = sum(a) / n; mb = sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    va = sum((x - ma) ** 2 for x in a); vb = sum((y - mb) ** 2 for y in b)
    return cov / math.sqrt(va * vb)


def main():
    out = {}
    # family-level pooled
    fam_tab = {}
    for fam in sorted(set(cat.values())):
        fi = [i for i in ids if cat[i] == fam]
        fam_tab[fam] = {"kind": kind[fi[0]], "n_inst": len(fi)}
        for cfg in ["B0", "B2", "B3", "B4"]:
            vals = [sem(mk, src, cfg, i) for mk, src, _ in MODELS for i in fi]
            fam_tab[fam][cfg] = round(sum(vals) / len(vals), 3)
    out["family_level_pooled"] = fam_tab
    # strong-model silent-semantic breakdown
    strong = {}
    for fam in ["wrong-gate-or-param", "endianness-bit-order", "missing-measurement"]:
        fi = [i for i in ids if cat[i] == fam]
        strong[fam] = {nm: {c: f"{sum(sem(mk, 'frozen', c, i) for i in fi)}/{len(fi)}"
                            for c in ["B0", "B2", "B3", "B4"]}
                       for nm, mk in [("Sonnet", SON), ("Haiku", HAI)]}
    out["strong_model_family"] = strong
    # stratified by kind
    strat = {}
    for kd in ["crash", "semantic"]:
        fi = [i for i in ids if kind[i] == kd]
        b2 = [sem(mk, src, "B2", i) for mk, src, _ in MODELS for i in fi]
        b0 = [sem(mk, src, "B0", i) for mk, src, _ in MODELS for i in fi]
        d, lo, hi = paired_bootstrap_diff(b2, b0); w, l, p = mcnemar(b0, b2)
        strat[kd] = {"n": len(b2), "diff": round(d, 3), "ci": [round(lo, 3), round(hi, 3)],
                     "mcnemar_p": round(p, 4), "wins": w, "losses": l}
    out["stratified_by_kind"] = strat
    # capability correlation
    rhos = [r for *_, r in MODELS]
    gains = [sum(sem(mk, src, "B2", i) for i in ids) / len(ids) - sum(sem(mk, src, "B0", i) for i in ids) / len(ids)
             for mk, src, _ in MODELS]
    rk = sorted(rhos); gk = sorted(gains)
    out["capability_corr"] = {"pearson": round(pearson(rhos, gains), 3),
                              "spearman": round(pearson([rk.index(x) for x in rhos], [gk.index(x) for x in gains]), 3),
                              "n_models": len(MODELS)}
    # expanded negatives (uses cached LLM-detector calls)
    try:
        from lrq.detectors import detect
        from lrq.repair import llm_detect
        bench = json.load(open(os.path.join(ROOT, "benchmark", "benchmark.json")))
        negs = [("clean-" + b["id"], b["buggy"]) for b in bench if b["kind"] == "clean"] + \
               [("fixed-" + b["id"], b["fixed"]) for b in bench if b["kind"] != "clean"]
        ufp = 0
        for _, src in negs:
            if {f.category for f in detect(src)} | {f.category for f in llm_detect(src, model=SON)}:
                ufp += 1
        out["expanded_negatives"] = {"n_negatives": len(negs), "matched_fixes": 32, "clean_idioms": 8,
                                     "union_false_positives": ufp, "precision": round(1 - ufp / len(negs), 3)}
    except Exception as e:
        out["expanded_negatives"] = {"error": str(e)[:120]}
    json.dump(out, open(f"{R}/revision_analyses.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
