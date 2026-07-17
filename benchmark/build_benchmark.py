"""Validate every benchmark instance by execution and emit benchmark.json.

Contract checked per instance:
  crash    : buggy -> error/timeout ; fixed -> ok
  semantic : buggy -> ok but RESULT != fixed ; fixed -> ok
  smell    : buggy -> ok but RESULT != fixed ; fixed -> ok
  clean    : buggy == fixed -> ok

Any violation is printed and the instance is marked invalid (excluded).
The emitted benchmark.json stores the serialised reference RESULT of each
fixed program so downstream scoring never recomputes it.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from lrq.harness import run_script, results_match  # noqa: E402
from instances import INSTANCES  # noqa: E402


def validate(inst: dict) -> tuple[bool, str, dict | None]:
    kind = inst["kind"]
    fixed = run_script(inst["fixed"])
    if fixed.status != "ok":
        return False, f"fixed did not run: {fixed.error}", None
    ref = fixed.result

    buggy = run_script(inst["buggy"])
    if kind == "clean":
        if buggy.status != "ok":
            return False, f"clean buggy did not run: {buggy.error}", None
        return True, "ok", ref
    if kind == "crash":
        if buggy.status == "ok":
            return False, "crash bug unexpectedly ran without error", None
        return True, "ok", ref
    # semantic / smell
    if buggy.status != "ok":
        return False, f"semantic bug crashed (expected wrong output): {buggy.error}", None
    match, dist = results_match(ref, buggy.result)
    if match:
        return False, f"semantic bug matches fixed (dist={dist:.4f}); not a real defect", None
    return True, f"ok (buggy vs fixed dist={dist:.4f})", ref


def main():
    out = []
    bad = 0
    for inst in INSTANCES:
        ok, msg, ref = validate(inst)
        flag = "OK " if ok else "BAD"
        print(f"[{flag}] {inst['id']:24s} {inst['category']:26s} {msg}")
        if ok:
            rec = dict(inst)
            rec["reference_result"] = ref
            out.append(rec)
        else:
            bad += 1
    path = os.path.join(HERE, "benchmark.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n{len(out)} valid / {len(INSTANCES)} total ({bad} excluded) -> {path}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
