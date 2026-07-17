"""Execution + semantic-equivalence harness for the LintRepair-RAG-Q pilot.

A benchmark program is a self-contained Python/Qiskit script that, on a
successful run, binds a module-level variable ``RESULT`` to its observable
output. ``RESULT`` is one of:

* ``dict[str, int]``  -- measurement counts (a sampling program), or
* a ``qiskit.quantum_info.Statevector`` / 1-D complex iterable -- a
  deterministic state (a no-measurement program).

We run each script in a *fresh subprocess* (so a crashing or hanging program
cannot take down the harness), serialise ``RESULT`` to JSON over a sentinel
line, and compare two results semantically:

* counts vs counts  -> total variation distance (TVD)
* state vs state    -> fidelity ``|<a|b>|^2`` (global-phase invariant)

This is the *oracle* behind "semantic preservation": a candidate repair is
accepted only if it runs AND its RESULT matches the human-fixed reference.
"""
from __future__ import annotations

import base64
import json
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass

SENTINEL = "@@LRQ_RESULT@@"

# Appended to every script so it self-serialises RESULT (or reports it is missing).
_EPILOGUE = f"""
import json as _json, base64 as _b64
def _lrq_dump():
    try:
        _r = RESULT  # noqa: F821
    except NameError:
        return {{"type": "missing"}}
    # counts dict
    if isinstance(_r, dict):
        return {{"type": "counts", "data": {{str(k): int(v) for k, v in _r.items()}}}}
    # qiskit Statevector or array-like of complex amplitudes
    try:
        import numpy as _np
        _arr = _np.asarray(getattr(_r, "data", _r)).ravel()
        return {{"type": "state", "data": [[float(_c.real), float(_c.imag)] for _c in _arr]}}
    except Exception as _e:  # pragma: no cover
        return {{"type": "unknown", "repr": repr(_r)[:200]}}
print("{SENTINEL}" + _b64.b64encode(_json.dumps(_lrq_dump()).encode()).decode())
"""


@dataclass
class RunOutcome:
    status: str           # "ok" | "error" | "timeout"
    result: dict | None   # serialised RESULT payload when status == "ok"
    error: str            # last traceback line / message when status != "ok"
    raw: str = ""         # full stderr (for the repair feedback loop)


def run_script(source: str, python: str | None = None, timeout: int = 60) -> RunOutcome:
    """Execute *source* in a subprocess and capture its serialised RESULT."""
    python = python or sys.executable
    full = source + "\n" + _EPILOGUE
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(full)
        path = fh.name
    try:
        env = dict(os.environ)
        env.setdefault("OMP_NUM_THREADS", "1")
        proc = subprocess.run(
            [python, path],
            capture_output=True, text=True, timeout=timeout, env=env,
        )
    except subprocess.TimeoutExpired:
        os.unlink(path)
        return RunOutcome("timeout", None, f"timed out after {timeout}s")
    os.unlink(path)

    payload = None
    for line in proc.stdout.splitlines():
        if line.startswith(SENTINEL):
            payload = json.loads(base64.b64decode(line[len(SENTINEL):]).decode())
            break

    if proc.returncode != 0 or payload is None or payload.get("type") in (None, "missing", "unknown"):
        err = (proc.stderr or "").strip()
        last = err.splitlines()[-1] if err else f"exit={proc.returncode}, no RESULT bound"
        return RunOutcome("error", None, last, raw=err)
    return RunOutcome("ok", payload, "", raw=proc.stderr.strip())


def _counts_tvd(a: dict, b: dict) -> float:
    ka = sum(a.values()) or 1
    kb = sum(b.values()) or 1
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0) / ka - b.get(k, 0) / kb) for k in keys)


def _state_fidelity(a: list, b: list) -> float:
    if len(a) != len(b):
        return 0.0
    # <a|b> with complex conjugate on a
    re = im = 0.0
    na = nb = 0.0
    for (ar, ai), (br, bi) in zip(a, b):
        re += ar * br + ai * bi
        im += ar * bi - ai * br
        na += ar * ar + ai * ai
        nb += br * br + bi * bi
    if na == 0 or nb == 0:
        return 0.0
    return (re * re + im * im) / (na * nb)


def results_match(ref: dict, cand: dict, tvd_tol: float = 0.08, fid_tol: float = 0.99) -> tuple[bool, float]:
    """Return (match, distance_metric) comparing two serialised RESULT payloads."""
    if ref is None or cand is None:
        return False, 1.0
    if ref["type"] != cand["type"]:
        return False, 1.0
    if ref["type"] == "counts":
        d = _counts_tvd(ref["data"], cand["data"])
        return d <= tvd_tol, d
    if ref["type"] == "state":
        f = _state_fidelity(ref["data"], cand["data"])
        return f >= fid_tol, f
    return False, 1.0
