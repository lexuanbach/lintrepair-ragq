"""Partial correlation of base repair competence with grounding's effect on the
quantum-semantic subset, controlling for that subset's unguided accuracy.

This is the analysis behind the paper's "-0.569 after adjusting for headroom".
It is written to be fully reproducible from the released per-instance records, so
every choice that affects the number is stated explicitly:

  unit of analysis   one model (n = 18); the semantic subset has only 9 programs,
                     so we resample models, not programs
  rho-hat            exact k/32, where k is the number of B0 patches that RUN,
                     recomputed from the records rather than read from a rounded field
  gain               pass@1(B2) - pass@1(B0) over the 9 quantum-semantic defects
  control            pass@1(B0) over the same 9 defects (the headroom term)
  estimator          Pearson partial correlation r(rho, gain | B0)
  bootstrap          10,000 resamples of models with replacement, seed 0
  interval           percentile, 2.5% and 97.5%
  degenerate draws   a resample with zero variance in any input is discarded and
                     the number discarded is reported

Run:  python experiments/run_partial_correlation.py
Out:  experiments/results/partial_correlation.json
"""
from __future__ import annotations

import json
import math
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
API_FAMILIES = {"deprecated-execute-api", "removed-aer-import",
                "deprecated-bind-parameters"}
N_BOOT = 10_000
SEED = 0


def _load_pairs():
    """model -> {instance_id: {"B0": {...}, "B2": {...}}} over the 32 defects."""
    bench = {r["id"]: r for r in json.load(open(os.path.join(RES, "records.json")))}
    cat = {i: r["category"] for i, r in bench.items()}
    kind = {i: r["kind"] for i, r in bench.items()}
    series: dict[str, dict] = {}
    for r in bench.values():
        if r["kind"] == "clean":
            continue
        for model, cfg in r["repair"].items():
            series.setdefault(model, {})[r["id"]] = {
                b: {"sem": bool(cfg[b]["pass1"].get("sem")),
                    "ran": bool(cfg[b]["pass1"].get("ran"))}
                for b in ("B0", "B2")
            }
    extra = json.load(open(os.path.join(RES, "models_extra.json")))
    for model, d in extra.items():
        for rec in d["records"]:
            series.setdefault(model, {})[rec["id"]] = {
                b: {"sem": bool(rec[b]["sem"]), "ran": bool(rec[b].get("ran"))}
                for b in ("B0", "B2")
            }
    return series, cat, kind


def _pearson(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    if sx == 0 or sy == 0:
        raise ZeroDivisionError("zero variance")
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


def _partial(x, y, z):
    """r(x, y | z), the Pearson correlation of x and y after removing z."""
    rxy, rxz, ryz = _pearson(x, y), _pearson(x, z), _pearson(y, z)
    denom = math.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    if denom == 0:
        raise ZeroDivisionError("degenerate control")
    return (rxy - rxz * ryz) / denom


def main() -> None:
    series, cat, kind = _load_pairs()
    models = sorted(series)
    rho, gain, base = [], [], []
    for m in models:
        d = series[m]
        ran = sum(1 for i in d if d[i]["B0"]["ran"])
        rho.append(ran / len(d))                      # exact k/32
        sem = [i for i in d if cat[i] not in API_FAMILIES and kind[i] == "semantic"]
        b0 = sum(d[i]["B0"]["sem"] for i in sem) / len(sem)
        b2 = sum(d[i]["B2"]["sem"] for i in sem) / len(sem)
        base.append(b0)
        gain.append(b2 - b0)

    point = _partial(rho, gain, base)

    rng = random.Random(SEED)
    draws, degenerate = [], 0
    n = len(models)
    for _ in range(N_BOOT):
        idx = [rng.randrange(n) for _ in range(n)]
        try:
            draws.append(_partial([rho[i] for i in idx],
                                  [gain[i] for i in idx],
                                  [base[i] for i in idx]))
        except ZeroDivisionError:
            degenerate += 1
    draws.sort()
    lo = draws[int(0.025 * len(draws))]
    hi = draws[int(0.975 * len(draws))]

    out = {
        "_provenance": "experiments/run_partial_correlation.py, recomputed from "
                       "records.json and models_extra.json; rho-hat is exact k/32.",
        "n_models": n,
        "n_semantic_programs": 9,
        "resampling_unit": "model",
        "n_boot": N_BOOT,
        "seed": SEED,
        "interval_rule": "percentile (2.5%, 97.5%)",
        "degenerate_resamples_discarded": degenerate,
        "corr_rho_base": _pearson(rho, base),
        "corr_base_gain": _pearson(base, gain),
        "corr_rho_gain": _pearson(rho, gain),
        "partial_corr_rho_gain_given_base": point,
        "ci95": [lo, hi],
    }
    path = os.path.join(RES, "partial_correlation.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"partial r = {point:.6f}   95% CI [{lo:.3f}, {hi:.3f}]   "
          f"({degenerate} degenerate draws discarded)")
    print(f"written: {path}")


if __name__ == "__main__":
    main()
