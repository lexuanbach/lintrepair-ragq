# LintRepair-RAG-Q + Q-Defects40 — reproducible artifact

Artifact for *"When Grounding Hurts: Retrieval and Validation for LLM Repair of
Quantum Programs"* (ICECCS 2026, anonymized for review).

Every number, table, and figure in the paper is produced by the scripts here over
**real executions**; nothing is hand-entered. The repair study spans **18 models**
across four providers (OpenAI, Google, Anthropic, and eleven open-weight models via
Ollama), 3B–72B, over the full capability range. The headline finding is a
**capability-governed crossover**: retrieval-grounding plus execution validation
*rescues* weak models but *anchors* the strongest, and prompt-level debiasing is
itself capability-governed — the only variant that helps everywhere does so by
leaking the benchmark's defect taxonomy.

The offline reproduction below runs entirely from the frozen result files and needs
**no API keys, no network, and not the response cache**. The raw LLM response cache
(~10.9k entries) is distributed separately as the `llm_cache.tar.gz` **release asset**
(see the repo's Releases) to keep the clone lean; download it into this directory and
run `tar xzf llm_cache.tar.gz` (it unpacks to `.llm_cache/`, or let `reproduce.sh` do
it) only if you want to **re-run** model calls with cache hits instead of live API
calls.

## Paper ↔ artifact map

Regenerate the single canonical numbers file with
`python experiments/make_paper_numbers.py` → `experiments/results/paper_numbers.json`
(it *reads*, never recomputes). Each paper claim maps to a result file and the
script that writes it:

| Paper element | Numbers | Result file | Written by |
|---|---|---|---|
| **RQ1 detection** (`tab:detection`) | static 0.719 / LLM 0.781 / union 0.812 recall @ 1.000 precision, 0 FP | `summary_FROZEN.json` (`detection`) | `run_all.py` |
| LintQ head-to-head | 0.000 (high-prec) / 0.344 (all checkers) | `lintq_comparison.json` | `experiments/lintq/score_lintq.py` |
| Detector-config robustness (Sonnet/Haiku) | B2static / B2noassert repair pass@1 | `detector_sensitivity.json` | `run_detector_sensitivity.py` |
| **RQ2 crossover** (`tab:cross` + `fig:crossover`, 18 models) | per-model B0–B4, ρ̂; pooled +0.160 [.109,.214], 168/76 | `cross_model.json` (`rows`,`pooled`) | `aggregate_models.py` |
| Interaction (Fig 4 line) | slope −0.47, 95% CI [−0.68,−0.19], R²=0.40, crossing ρ̂≈0.84 | `interaction_and_losses.json` | `run_revision_interaction.py` |
| Stratified by kind | crash +0.258 [.196,.321]; silent −0.093 [−.173,−.012] | `revision_analyses.json` (`stratified_by_kind`) | `run_revision_analyses.py` |
| Loss breakdown (153/15 wins, 48/76 high-cap) | wins/losses by kind × capability | `interaction_and_losses.json` | `run_revision_interaction.py` |
| Per-family pooled (B0→B2) | removed-aer 0.32→0.81, … | `revision_analyses.json` (`family_level_pooled`) | `run_revision_analyses.py` |
| Capability correlation | Pearson −0.629, Spearman −0.543 | `revision_analyses.json` (`capability_corr`) | `run_revision_analyses.py` |
| Migration-script baseline | 10/32, API-removal only | `scripted_baseline.json` | `run_scripted_baseline.py` |
| **RQ3** debiasing | B3−B2 +0.012 p=.48; B4−B2 −0.030 p=.04; B3−B4 +0.042 p<.001 | `cross_model.json` (`pooled`) | `aggregate_models.py` |
| **§ Dissection** matched controls | B0filler drop (Sonnet 9→6, Haiku 7→3) | `anchoring_controls.json` | `run_anchoring_controls.py` |
| pass@5 robustness | Sonnet 9/9 vs 6/9 | `passk.json` | `run_passk.py` |
| Powered n=115 silent set | Sonnet 42→0, open weights collapse | `mutation_crossover.json` | `run_mutation_crossover.py` |
| Detector-hit audit | static 23/23, LLM 25/25, union 26/26; 0 FP | `interaction_and_losses.json` (`detector_audit`) | `run_revision_interaction.py` |
| Retrieval-budget dial | Sonnet best at k=2; GPT-4o-mini monotone | `retrieval_sensitivity.json` | `run_retrieval_sensitivity.py` |
| **§ Replication** Bugs4Q | 2/42 vs 19/42; silent anchoring 4/10→0/10 | `bugs4q_summary.json` | `run_bugs4q.py` |
| Cross-SDK Cirq (negative) | 10-program Cirq, no interference (Sonnet/Haiku 10/10) | `cirqdefects.json` | `run_cirqdefects.py` (from `benchmark/cirq_defects.py`) |
| Oracle robustness | labels invariant over TVD∈[0.02,0.20] | `oracle_sensitivity.json` | `run_oracle_sensitivity.py` |
| Cost | latency per call | `latency_percall.json` | `run_all.py` |

## Layout

- `benchmark/instances.py` — the 40 Q-Defects40 programs (32 defective across 8
  families + 8 clean), each with buggy and human-fixed versions and labels.
- `benchmark/build_benchmark.py` — **validates by execution** that crash defects
  crash, semantic defects run-but-diverge, and fixes/clean run; emits
  `benchmark/benchmark.json` with the reference observable per program.
- `benchmark/cirq_defects.py` — the 10-program Cirq cross-SDK probe.
- `lrq/harness.py` — subprocess execution + semantic oracle (TVD over counts,
  state fidelity for measurement-free circuits).
- `lrq/detectors.py` — static AST/pattern linter (rule-based detector).
- `lrq/retrieval.py` — knowledge base (15 general defect-family docs, no instance
  answers) + dependency-free TF-IDF retriever.
- `lrq/repair.py` — LLM detector and the five repair configs + validator loop
  (see below).
- `lrq/llm.py` — multi-provider wrapper (Anthropic / OpenAI / Google Gemini /
  local Ollama) with on-disk caching and per-call latency/token accounting.
- `lrq/stats.py` — Wilson intervals, paired bootstrap, exact McNemar.
- `experiments/` — one runner per experiment (see the map above) plus:
  - `run_all.py` — frozen pilot: detection + repair (B0–B4) on Sonnet 4.6 and
    Haiku 4.5 → `results/summary_FROZEN.json`, `records_FROZEN.json`.
  - `run_models.py` — the other 16 backbones; reconstructs detection findings from
    the frozen records (no Anthropic call, only the repair backbone varies) →
    `results/models_extra.json`.
  - `aggregate_models.py` — unifies frozen + extra into the 18-model table and
    pooled tests → `results/cross_model.json`. No API calls.
  - `make_paper_numbers.py` — assembles the canonical `results/paper_numbers.json`.
- `reproduce.sh` — one-command, cache-only replay of all of the above, ending in a
  paper-vs-artifact consistency assertion (fail-closed).
- `check_consistency.py` — asserts every headline manuscript number equals
  `paper_numbers.json` (exits non-zero on any drift).
- `make_manifest.py` — writes `MANIFEST.json` (SHA-256 + size of all 70 source/result
  files); `--check` re-verifies it for tamper evidence.

## Repair configurations (B0–B4)

- **B0** plain — no grounding (the unguided baseline).
- **B1** RAG — findings + retrieval, no validator loop.
- **B2** naive-grounded — findings + retrieval + execution-validator loop.
- **B3** named-debiased — B2 plus a caveat that *names* the silent defect families
  (leaks the taxonomy; reported only as an ablation control).
- **B4** generic-debiased — B2 plus a generic "analyser may be incomplete" caveat
  (the deployable variant).

## Reproduce

**Every reported number, in one cache-only command (no API, no network):**

```bash
python3.10 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
./reproduce.sh          # fails closed if a required result file is missing
```

`reproduce.sh` re-aggregates the 18-model table, the revision analyses, the
interaction/loss breakdown, and the canonical `results/paper_numbers.json`. The
individual steps (and the cold-run paths) are:

```bash
python benchmark/build_benchmark.py     # validate the benchmark by execution (no API)

# Frozen pilot (Sonnet/Haiku): cached -> regenerates the pilot tables for free.
export ANTHROPIC_API_KEY=...             # only for a cold run
python experiments/run_all.py

# The other 16 backbones (pass the list explicitly). Cached -> free; a cold run
# needs OpenAI/Gemini keys and a local Ollama serving the open-weight models.
python experiments/run_models.py \
  gpt-4o gpt-4o-mini gemini-2.5-pro gemini-2.5-flash gemini-2.5-flash-lite \
  qwen2.5:7b qwen2.5:14b qwen2.5:72b qwen2.5-coder:7b qwen2.5-coder:32b \
  llama3.1:8b llama3.2:3b llama3.3:70b gemma2:9b mistral:7b command-r:35b
python experiments/aggregate_models.py   # frozen + extra -> results/cross_model.json (no API)

# Revision analyses and the single canonical numbers file (all no-API, from cache):
python experiments/run_revision_analyses.py
python experiments/run_revision_interaction.py
python experiments/make_paper_numbers.py # -> results/paper_numbers.json
```

## Self-contained

Everything needed to regenerate **every reported number** (assembled into
`results/paper_numbers.json` by `make_paper_numbers.py`, and the source of every
table/figure value) is inside this folder: the benchmark (`benchmark/`), the
toolchain (`lrq/`), the run scripts and frozen per-experiment results
(`experiments/`, `experiments/results/`), and the on-disk response cache
(`.llm_cache/`). Manuscript table/figure values are transcribed from these files;
we do not currently auto-generate the LaTeX from the JSON. No file is read from outside the
artifact root, and no sibling project is imported. The core path
(`build_benchmark.py` → `run_all.py`/`run_models.py` → `aggregate_models.py` →
`run_revision_*.py` → `make_paper_numbers.py`) runs from cache with **no API keys
and no network** — only `pip install -r requirements.txt`.

**Cold-run API keys** are read from environment variables
(`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`); a keys *directory* is an
optional fallback (`LRQ_KEYS_DIR`). Open-weight models use a local Ollama endpoint
(`http://localhost:11434`).

**Auxiliary experiments that wrap external tools** are the only parts not runnable
from this folder alone; their **results are frozen** in `experiments/results/`, so
they are not needed to reproduce the paper:

- `run_bugs4q.py` / `mine_realbugs.py` — re-executing Bugs4Q bugs needs a legacy
  Qiskit 0.45 interpreter (`BUGS4Q_LEGACY_PY`); the Bugs4Q set is bundled
  (`experiments/bugs4q_real.json`). Frozen: `results/bugs4q_summary.json`.
- `lintq/score_lintq.py` — scores a SARIF produced by CodeQL + the LintQ queries
  (external); ground-truth labels are bundled (`experiments/lintq/labels.json`).
  Frozen: `results/lintq_comparison.json`.
- `run_llm_failed_gen.py` — uses the external QuanBench+ prompt set
  (`QUANBENCH_DIR`). Frozen: `results/llm_failed_dataset.json`.

## Versions & models

Python 3.10, Qiskit 2.4.2, Qiskit-Aer 0.17.2. Greedy decoding (temperature 0);
8192 shots with fixed simulation seeds; oracle TVD ≤ 0.08, fidelity ≥ 0.99;
validator budget m = 3; retrieval top-k = 4.

**18 models (4 providers).** OpenAI `gpt-4o`, `gpt-4o-mini`; Google
`gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-2.5-flash-lite`; Anthropic
`claude-sonnet-4-6`, `claude-haiku-4-5`; open-weight via Ollama `qwen2.5:7b/14b/72b`,
`qwen2.5-coder:7b/32b`, `llama3.1:8b`, `llama3.2:3b`, `llama3.3:70b`, `gemma2:9b`,
`mistral:7b`, `command-r:35b`. Anthropic credits are exhausted, so `claude-*`
reproduces from cache only. Each cache entry stores the response text, token counts,
model id, and per-call latency; it does not currently record a wall-clock timestamp
or an immutable provider revision, so exact hosted-model snapshots are not pinned.
