# Supplement — *When Grounding Hurts* (ICECCS 2026)

Reviewer-facing detail that does not fit the 20-page main paper. Every item points
to the exact source/result file in this artifact; nothing here is new evidence.

## S1. Repair configurations B0–B4 (prompt framing)

Defined in [`lrq/repair.py`](lrq/repair.py) (the exact prompt strings are in that
file; here is the contract):

| Config | Findings | Retrieval | Validator loop | Cleanliness assertion when detector silent | Debias caveat |
|---|---|---|---|---|---|
| **B0** plain | — | — | — | — | — |
| **B1** RAG | union | ✓ | — | "no structural defects" | — |
| **B2** naive | union | ✓ | ✓ | "no structural defects" | — |
| **B3** named | union | ✓ | ✓ | never | *names* the silent families (wrong/forgotten gate, swapped control/target, degrees-vs-radians, endianness) |
| **B4** generic | union | ✓ | ✓ | never | generic "analyser is incomplete, may miss SILENT logic errors; verify independently" — **no family names** |

`B4` is the deployable variant; `B3` is an ablation control whose only difference
from `B4` is naming the benchmark's defect families, so `B3−B4` isolates taxonomy
leakage. `B0filler` (used in the dissection) supplies the same retrieved documents
as neutral background with **no findings and no cleanliness claim**, isolating
context-distraction from grounding content.

## S2. Statistical methods

Implemented in [`lrq/stats.py`](lrq/stats.py): `wilson` (Wilson 95% intervals),
`paired_bootstrap_diff` (10⁴ resamples, fixed LCG seed, deterministic), `mcnemar`
(exact two-sided). Cluster-aware inference is in
[`experiments/run_clustered_stats.py`](experiments/run_clustered_stats.py): a
**program-clustered bootstrap** (resampling whole programs) and a **mixed-effects
logistic** with crossed random intercepts for program and model. Significance is
claimed only where the bootstrap CI excludes zero **and** exact McNemar agrees.
Two pooled claims near *p*≈0.04 (silent-semantic harm; B4 erosion) do not survive
Bonferroni over ~7 tests and are reported as corroborated by the powered set, not
established (see the paper's Statistical-conclusion threat).

## S3. Formal model scope

Proposition 2, `R(ρ,m)=1−(1−ρ)^m`, assumes **iid** retries and is used as a
*reference* model. The implemented loop ([`lrq/repair.py`](lrq/repair.py)) feeds
each runtime error into the next prompt, so real retries are **adaptive**; RQ4
tests the qualitative prediction (value scaling with 1−ρ), not the exact form. The
grounding decomposition (Eq. 1) is the exact identity
`Δ = R_g s_g − R_u s_u = (R_g−R_u) s_u + R_g (s_g−s_u)`. `R_g ≥ R_u` is a
scope-bounded assumption for API-crash defects (confirmed for the API-removal
families, not where no API is missing).

## S4. Q-Defects40 and the oracle

40 self-contained Qiskit 2.4.2 programs — 32 defective (8 families × 4) + 8 clean —
in [`benchmark/instances.py`](benchmark/instances.py). Families and provenance
(M = Qiskit 1.0 migration guide, L = LintQ/Smelly-Eight, B = Bugs4Q-style silent):
`deprecated-execute-api`, `removed-aer-import`, `deprecated-bind-parameters`,
`qubit-index-oob`, `clbit-size-mismatch` (all crash, static-visible);
`missing-measurement` (mixed); `wrong-gate-or-param`, `endianness-bit-order`
(silent, no syntactic signature). The oracle (TVD ≤ 0.08 at 8192 shots / fidelity
≥ 0.99) is in [`lrq/harness.py`](lrq/harness.py).
[`benchmark/build_benchmark.py`](benchmark/build_benchmark.py) **validates by
execution** that every crash defect raises, every silent defect runs-but-diverges
(median TVD 0.50), and every fix/clean program runs; it rejected one proposed
endianness instance that computed the correct value.

## S5. Powered silent set (115 mutants)

Constructed by [`experiments/run_mutation_gen.py`](experiments/run_mutation_gen.py):
silent-leaning operators — gate substitution (`GATE_SUBS`:
cx↔cz, cx→cy, h→x, x→y, s→t, cx→ch), swapping the two qubit args of a 2-qubit gate,
and rotation-angle perturbation — applied one at a time to the correct programs.
**Every mutant is execution-validated** (runs, diverges from the reference) before
inclusion; non-diverging mutants are discarded. Result:
[`results/mutation_crossover.json`](experiments/results/mutation_crossover.json).
These are synthetic and used only as a power/robustness check.

## S6. Detectors

Static analyser: AST/pattern rules in [`lrq/detectors.py`](lrq/detectors.py).
RAG-LLM detector: [`lrq/repair.py`](lrq/repair.py), **fixed to one model (Sonnet)**
across all repair experiments, so findings are identical regardless of repair
backbone. Detection audit (every flagged program names its true family; static
23/23, LLM 25/25, union 26/26; 0 FP) in
[`results/interaction_and_losses.json`](experiments/results/interaction_and_losses.json).
**Known gap:** we do *not* release a weaker-detector (Haiku) recall control; that
experiment was not run as a detector, so the paper makes no Haiku-detector recall
claim. Detector generality is stated as the top open threat.

## S7. Retrieval

Dependency-free TF–IDF over 15 general defect-family documents (no instance
answers), top-*k* = 4, in [`lrq/retrieval.py`](lrq/retrieval.py). Budget sensitivity
(Sonnet best at k=2; GPT-4o-mini monotone in k; BM25 ≈ TF–IDF) in
[`results/retrieval_sensitivity.json`](experiments/results/retrieval_sensitivity.json).

## S8. Cross-SDK Cirq reconciliation

The paper reports the **10-program** Cirq probe
([`benchmark/cirq_defects.py`](benchmark/cirq_defects.py) →
[`experiments/run_cirqdefects.py`](experiments/run_cirqdefects.py) →
[`results/cirqdefects.json`](experiments/results/cirqdefects.json)): Sonnet and
Haiku repair 10/10 with grounding neutral, i.e. no interference. An earlier
**8-program** pilot ([`results/crosssdk_cirq.json`](experiments/results/crosssdk_cirq.json))
is superseded and not used by the canonical assembler.

## S9. External-tool experiments (provenance)

Results are **frozen**; re-execution needs external tools (disclosed, not bundled):

- **Bugs4Q** — bugs execute under legacy **Qiskit 0.45** (`BUGS4Q_LEGACY_PY`); the
  bug set is bundled ([`experiments/bugs4q_real.json`](experiments/bugs4q_real.json)).
  Frozen: [`results/bugs4q_summary.json`](experiments/results/bugs4q_summary.json).
- **LintQ** — SARIF produced by CodeQL 2.11.2 + the LintQ queries; labels bundled
  ([`experiments/lintq/labels.json`](experiments/lintq/labels.json)), scorer
  [`experiments/lintq/score_lintq.py`](experiments/lintq/score_lintq.py). Frozen:
  [`results/lintq_comparison.json`](experiments/results/lintq_comparison.json).
- **QuanBench+** — external prompt set (`QUANBENCH_DIR`). Frozen:
  [`results/llm_failed_dataset.json`](experiments/results/llm_failed_dataset.json).

## S10. Reproduction & provenance

`./reproduce.sh` runs the entire cache-only pipeline (no API/network) and ends in
`check_consistency.py`, which asserts every headline number equals
`results/paper_numbers.json` (fail-closed). `make_manifest.py` writes `MANIFEST.json`
(SHA-256 + size of all source/result files) and re-verifies it with `--check`. Model
snapshots: hosted-model revisions are **not** pinned (see the README); cache entries
store text, tokens, model id, and latency but no wall-clock date. For a byte-exact
environment, generate a `pip freeze` lock in the packaging environment.
