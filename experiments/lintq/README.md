# LintQ head-to-head on Q-Defects40

Real run of **LintQ** (FSE'24, `sola-st/LintQ` @ 481ca24) against our 40 programs,
producing the comparison in `../results/lintq_comparison.json` and the detection
table in the paper. No numbers are hand-entered; both SARIFs are kept here.

## Reproduce
```bash
# 1. Build LintQ's CodeQL image (bundles CodeQL 2.11.2 + the qlint packs)
git clone https://github.com/sola-st/LintQ && cd LintQ
python automation_scripts/create_docker_image_locally.py     # -> image: lintq

# 2. Export the 40 buggy programs as .py (one per benchmark instance)
python - <<'PY'
import json, os
b=json.load(open('../benchmark/benchmark.json')); d='work/files'; os.makedirs(d)
for x in b: open(f"{d}/{x['id']}.py","w").write(x['buggy'])
PY

# 3. Clean CodeQL DB of ONLY our 40 files (empty venv => no library extraction)
docker run --rm -v "$PWD/work:/work" lintq bash -lc '
  python3 -m venv /tmp/e && PATH=/tmp/e/bin:$PATH \
  codeql database create /work/clean_db --language=python --source-root /work/files --overwrite'

# 4a. LintQ default (recommended high-precision suite)  -> lintq_default.sarif
docker run --rm -v "$PWD/work:/work" -v "$PWD/LintQ.qls:/work/LintQ.qls" lintq \
  codeql database analyze /work/clean_db /work/LintQ.qls --format=sarifv2.1.0 \
  --output=/work/lintq_default.sarif --rerun
# 4b. LintQ-all (every LintQ-tagged checker)            -> lintq_all.sarif
docker run --rm -v "$PWD/work:/work" -v "$PWD/LintQ-all.qls:/work/LintQ-all.qls" lintq \
  codeql database analyze /work/clean_db /work/LintQ-all.qls --format=sarifv2.1.0 \
  --output=/work/lintq_all.sarif --rerun

# 5. Score program-level precision/recall vs labels
python score_lintq.py lintq_all.sarif
```

## Result
- **Model validity:** the pack's `ql-qc-size` (CircuitSize) query fires on our programs,
  confirming LintQ's Qiskit abstraction *modeled* the circuits (so non-detection is
  genuinely out-of-scope, not a parse failure).
- **LintQ default (high-precision):** recall **0.000** (0/32), 0 false positives.
- **LintQ all-checkers:** precision **0.786**, recall **0.344**, F1 **0.478** — from only two
  incidental structural checkers (`UnmeasurableQubits`, `ConstantClassicBit`) that also
  misfire on 3/8 clean programs, and that flag endianness-buggy and endianness-correct
  programs identically (i.e. not detecting the defect).

Compare with `\sys{}` union detection: precision 1.000, recall 0.812.
