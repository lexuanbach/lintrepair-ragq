#!/usr/bin/env bash
# One-command, cache-only reproduction of every reported number.
# Makes NO API call and needs NO network: it re-derives the canonical
# results/paper_numbers.json from the frozen per-experiment result files and the
# on-disk .llm_cache. Fails closed if a required result file is missing.
set -euo pipefail
cd "$(dirname "$0")"

# The frozen result files below are sufficient for offline reproduction. The raw LLM
# response cache (~10.9k entries) is a GitHub release asset, llm_cache.tar.gz; if you
# downloaded it here, unpack it so experiment RE-runs get cache hits, not live calls.
if [ ! -d .llm_cache ]; then
  for t in llm_cache.tar.gz .llm_cache.tar.gz; do
    [ -f "$t" ] && { echo "[0/5] extracting $t ..."; tar xzf "$t"; break; }
  done
fi

req() { [ -f "experiments/results/$1" ] || { echo "MISSING: results/$1 (cannot reproduce offline)"; exit 1; }; }
for f in records_FROZEN.json models_extra.json summary_FROZEN.json revision_analyses.json \
         lintq_comparison.json bugs4q_summary.json cirqdefects.json; do req "$f"; done

echo "[1/4] aggregate the 18-model cross-model table ...";      python3 experiments/aggregate_models.py        >/dev/null
echo "[2/4] revision analyses (stratified/family/capability) ..."; python3 experiments/run_revision_analyses.py >/dev/null
echo "[3/4] interaction regression + loss/detector-audit ...";  python3 experiments/run_revision_interaction.py >/dev/null
echo "[4/5] assemble canonical results/paper_numbers.json ...";  python3 experiments/make_paper_numbers.py
echo "[5/5] verify paper <-> artifact consistency ...";          python3 check_consistency.py

echo
echo "OK. Canonical numbers: experiments/results/paper_numbers.json"
echo "Provenance manifest: run 'python3 make_manifest.py' (verify: --check)."
