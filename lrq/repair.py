"""The LintRepair-RAG-Q agent and its baselines / ablations.

Configurations evaluated:
  B0  plain-LLM        : single-shot "fix this code"; no findings, no retrieval, no validator
  B1  RAG-LLM          : + static+LLM detection findings + retrieved KB docs; single shot
  B2  LintRepair-RAG-Q : B1 + execution validator loop (iterate on compile/run errors)

The agent's *internal* validator is execution-based: a candidate is accepted
when it parses, imports, and runs to completion binding RESULT (no exception).
It does NOT see the human-fixed reference -- semantic correctness against that
reference is measured externally by the evaluation harness. So the validator
loop directly helps crash defects (iterate until the program runs) while
silent-semantic defects depend on detection + retrieval + the model's reasoning.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

from .detectors import detect, Finding
from .retrieval import TfidfRetriever
from .llm import complete, last_latency

_RETRIEVER = TfidfRetriever()

TAXONOMY = [
    "deprecated-execute-api", "removed-aer-import", "deprecated-bind-parameters",
    "qubit-index-oob", "clbit-size-mismatch", "missing-measurement",
    "wrong-gate-or-param", "endianness-bit-order",
]

_CODE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    m = _CODE_RE.findall(text)
    if m:
        return max(m, key=len).strip()
    return text.strip()


# --------------------------------------------------------------------------
# Detection (LLM, retrieval-grounded)
# --------------------------------------------------------------------------
_DETECT_SYS = (
    "You are a static analyser for Qiskit quantum programs. You identify defects "
    "and report them as strict JSON. Be precise: report a category only if the code "
    "actually exhibits it."
)


def llm_detect(source: str, k: int = 4, model: str | None = None) -> list[Finding]:
    docs = _RETRIEVER.retrieve(source, k=k)
    ctx = "\n".join(f"- [{d.tag}] {d.text}" for d in docs)
    prompt = f"""Analyse this Qiskit program for defects.

Relevant knowledge:
{ctx}

Allowed categories: {", ".join(TAXONOMY)}.

Program:
```python
{source}
```

Return ONLY a JSON array. Each element: {{"category": <one allowed category>,
"line": <1-based int>, "reason": <short>}}. If there are no defects, return [].
"""
    txt = complete(_DETECT_SYS, prompt, temperature=0.0, max_tokens=700, model=model)
    try:
        arr = json.loads(txt[txt.index("["):txt.rindex("]") + 1])
    except Exception:
        return []
    out = []
    for e in arr:
        if isinstance(e, dict) and e.get("category") in TAXONOMY:
            out.append(Finding(e["category"], int(e.get("line", 0) or 0), str(e.get("reason", ""))[:120]))
    return out


# --------------------------------------------------------------------------
# Repair configurations
# --------------------------------------------------------------------------
_REPAIR_SYS = (
    "You are an expert Qiskit engineer. You repair a quantum program so it runs "
    "correctly on Qiskit 2.x with the qiskit_aer simulator, preserving the author's "
    "evident intent. You output the COMPLETE corrected program in a single ```python "
    "code block and nothing else. Keep the same observable behaviour variable RESULT."
)


@dataclass
class RepairResult:
    code: str
    explanation: str = ""
    iters: int = 1
    llm_calls: int = 0
    latency: float = 0.0
    findings: list = field(default_factory=list)
    trace: list = field(default_factory=list)


def plain_repair(source: str, sample: int = 0, model: str | None = None) -> RepairResult:
    """B0: no findings, no retrieval, no validator."""
    t0 = time.time()
    prompt = f"The following Qiskit program is buggy. Fix it.\n\n```python\n{source}\n```"
    txt = complete(_REPAIR_SYS, prompt, temperature=0.0 if sample == 0 else 0.6,
                   max_tokens=1500, sample=sample, model=model)
    return RepairResult(code=extract_code(txt), explanation=txt, iters=1, llm_calls=1,
                        latency=last_latency())


def _findings_block(findings: list[Finding], debias=False) -> str:
    """debias: False (naive) | True/'named' (enumerates silent families, B3) |
    'generic' (incomplete-analyser caveat with NO family names, B4 -- isolates
    anchoring-removal from answer-injection / family leakage)."""
    if not debias:
        if not findings:
            return "(static analysis reported no structural defects)"
        return "\n".join(f"- {f.category} at line {f.line}: {f.evidence}" for f in findings)
    if debias == "noassert":
        # detector-sensitivity control: same as naive but WITHOUT the cleanliness
        # assertion when nothing is flagged -- isolates whether the "no defects"
        # claim (rather than the missing info) drives anchoring.
        if not findings:
            return "(the analyser flagged no specific structural or API issues)"
        return "\n".join(f"- {f.category} at line {f.line}: {f.evidence}" for f in findings)
    if debias == "generic":
        header = ("Static+retrieval analysis flagged these CANDIDATE issues. The analyser "
                  "is incomplete and may miss SILENT logic errors that yield a wrong result "
                  "without raising an error. Do NOT assume the code is correct where the "
                  "analyser is silent -- independently verify that the circuit and its "
                  "classical post-processing implement the author's intended computation.")
    else:  # 'named' (B3): enumerates the silent failure modes
        header = ("Static+retrieval analysis flagged these CANDIDATE issues. The analyser "
                  "is incomplete: it can miss SILENT logic errors (a wrong or forgotten gate, "
                  "a swapped control/target, a rotation angle in degrees not radians, a "
                  "little-endian bit-order mistake in result post-processing). Do NOT assume "
                  "the code is correct where the analyser is silent -- independently verify the "
                  "circuit's intended behaviour and its classical post-processing.")
    body = ("\n".join(f"- {f.category} at line {f.line}: {f.evidence}" for f in findings)
            if findings else "- (no structural/API issue flagged; inspect the logic yourself)")
    return header + "\n" + body


def rag_repair(source: str, findings: list[Finding], sample: int = 0, k: int = 4,
               debias: bool = False, model: str | None = None) -> RepairResult:
    """B1: retrieval + findings, single shot, no validator."""
    t0 = time.time()
    docs = _RETRIEVER.retrieve(source + " " + " ".join(f.category for f in findings), k=k)
    ctx = "\n".join(f"- [{d.tag}] {d.text}" for d in docs)
    prompt = f"""Repair this Qiskit program.

Detected issues:
{_findings_block(findings, debias)}

Relevant knowledge (retrieved):
{ctx}

```python
{source}
```

Return the complete corrected program in one ```python block, then one line
"EXPLANATION:" followed by a one-sentence rationale."""
    txt = complete(_REPAIR_SYS, prompt, temperature=0.0 if sample == 0 else 0.6,
                   max_tokens=1600, sample=sample, model=model)
    code = extract_code(txt)
    expl = txt.split("EXPLANATION:")[-1].strip() if "EXPLANATION:" in txt else ""
    return RepairResult(code=code, explanation=expl, iters=1, llm_calls=1,
                        latency=last_latency(), findings=[f.category for f in findings])


def full_repair(source: str, findings: list[Finding], runner, sample: int = 0,
                k: int = 4, max_iters: int = 3, debias: bool = False,
                model: str | None = None) -> RepairResult:
    """B2: B1 + execution validator loop. `runner(code)->RunOutcome` validates."""
    t0 = time.time()
    docs = _RETRIEVER.retrieve(source + " " + " ".join(f.category for f in findings), k=k)
    ctx = "\n".join(f"- [{d.tag}] {d.text}" for d in docs)
    history = ""
    code, expl, calls, trace, lat = source, "", 0, [], 0.0
    for it in range(1, max_iters + 1):
        prompt = f"""Repair this Qiskit program so it runs correctly on Qiskit 2.x.

Detected issues:
{_findings_block(findings, debias)}

Relevant knowledge (retrieved):
{ctx}
{history}
```python
{source}
```

Return the complete corrected program in one ```python block, then one line
"EXPLANATION:" followed by a one-sentence rationale."""
        txt = complete(_REPAIR_SYS, prompt, temperature=0.0 if sample == 0 else 0.6,
                       max_tokens=1600, sample=sample * 10 + it, model=model)
        calls += 1
        lat += last_latency()
        code = extract_code(txt)
        expl = txt.split("EXPLANATION:")[-1].strip() if "EXPLANATION:" in txt else ""
        outcome = runner(code)
        trace.append({"iter": it, "status": outcome.status, "error": outcome.error[:160]})
        if outcome.status == "ok":
            break
        # validator feedback for the next iteration
        history = (f"\nYour previous attempt failed validation with this error:\n"
                   f"{outcome.error}\nFix it.\n")
    return RepairResult(code=code, explanation=expl, iters=it, llm_calls=calls,
                        latency=lat, findings=[f.category for f in findings],
                        trace=trace)


def all_findings(source: str) -> list[Finding]:
    """Union of static + LLM detection (de-duplicated by category)."""
    static = detect(source)
    llm = llm_detect(source)
    seen, out = set(), []
    for f in static + llm:
        if f.category not in seen:
            seen.add(f.category)
            out.append(f)
    return out
