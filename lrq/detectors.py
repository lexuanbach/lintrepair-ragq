"""Static rule-based detectors for the LintRepair-RAG-Q pilot.

A lightweight, source-level (AST + pattern) analyser in the spirit of LintQ:
it never executes the program.  It targets the *structural / API* defect
families that static analysis can soundly flag:

  deprecated-execute-api, removed-aer-import, deprecated-bind-parameters,
  qubit-index-oob, clbit-size-mismatch, missing-measurement

By construction it does **not** target the silent-semantic families
(wrong-gate-or-param, endianness-bit-order): those have no syntactic
signature, so a rule-based linter misses them.  That recall gap is the
empirical motivation for the retrieval-grounded LLM layer.

A finding is ``(category, line, evidence)``.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass


@dataclass(frozen=True)
class Finding:
    category: str
    line: int
    evidence: str


# method name -> positions of its *qubit* arguments (others are params/clbits)
_SINGLE_Q = {"h", "x", "y", "z", "s", "sdg", "t", "tdg", "id", "i", "sx", "reset"}
_TWO_Q = {"cx", "cz", "cy", "ch", "swap", "iswap", "dcx"}
_ROT_1Q = {"rx", "ry", "rz", "p", "u", "u1", "u2", "u3", "phase"}  # last arg(s) qubit
_CROT = {"crx", "cry", "crz", "cp", "cu1"}  # param first, then 2 qubits
_THREE_Q = {"ccx", "cswap", "mcx"}


def _const_int(node) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    return None


def _qubit_arg_indices(method: str, nargs: int) -> list[int]:
    if method in _SINGLE_Q:
        return [0]
    if method in _TWO_Q:
        return [0, 1]
    if method in _THREE_Q:
        return list(range(nargs))
    if method in _ROT_1Q:
        return list(range(1, nargs))      # angle(s) first, qubit(s) after
    if method in _CROT:
        return [1, 2]
    if method == "measure":
        return [0]                         # arg0 qubit, arg1 clbit
    return []


class _Analyzer(ast.NodeVisitor):
    def __init__(self, source: str):
        self.source = source
        self.findings: list[Finding] = []
        self.reg_sizes: dict[str, int] = {}     # QuantumRegister/ClassicalRegister var -> size
        self.reg_kind: dict[str, str] = {}       # var -> 'q' | 'c'
        self.circuits: dict[str, tuple[int | None, int | None]] = {}  # var -> (nq, nc)
        self.has_measure = False
        self.has_get_counts = False

    # --- imports ----------------------------------------------------------
    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module and node.module.startswith("qiskit"):
            for a in node.names:
                if a.name == "execute":
                    self.findings.append(Finding("deprecated-execute-api", node.lineno,
                                                 "from qiskit import execute"))
                if a.name == "Aer":
                    self.findings.append(Finding("removed-aer-import", node.lineno,
                                                 "from qiskit import Aer"))
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        # qiskit.execute / qiskit.Aer
        if isinstance(node.value, ast.Name) and node.value.id == "qiskit":
            if node.attr == "execute":
                self.findings.append(Finding("deprecated-execute-api", node.lineno, "qiskit.execute"))
            if node.attr == "Aer":
                self.findings.append(Finding("removed-aer-import", node.lineno, "qiskit.Aer"))
        self.generic_visit(node)

    # --- assignments: track registers and circuits ------------------------
    def visit_Assign(self, node: ast.Assign):
        if isinstance(node.value, ast.Call):
            callee = node.value.func
            name = callee.attr if isinstance(callee, ast.Attribute) else getattr(callee, "id", None)
            tgt = node.targets[0]
            var = tgt.id if isinstance(tgt, ast.Name) else None
            if name in ("QuantumRegister", "ClassicalRegister") and var:
                size = _const_int(node.value.args[0]) if node.value.args else None
                if size is not None:
                    self.reg_sizes[var] = size
                    self.reg_kind[var] = "q" if name == "QuantumRegister" else "c"
            if name == "QuantumCircuit" and var:
                self.circuits[var] = self._circuit_dims(node.value.args)
        self.generic_visit(node)

    def _circuit_dims(self, args) -> tuple[int | None, int | None]:
        nq = nc = None
        if len(args) >= 1:
            v = _const_int(args[0])
            if v is not None:
                nq = v
            elif isinstance(args[0], ast.Name) and self.reg_kind.get(args[0].id) == "q":
                nq = self.reg_sizes.get(args[0].id)
        if len(args) >= 2:
            v = _const_int(args[1])
            if v is not None:
                nc = v
            elif isinstance(args[1], ast.Name) and self.reg_kind.get(args[1].id) == "c":
                nc = self.reg_sizes.get(args[1].id)
        else:
            # QuantumCircuit(n) -> no classical bits unless a register was passed
            if nq is not None:
                nc = 0
        return nq, nc

    # --- calls: gates, measures, bind_parameters, get_counts --------------
    def visit_Call(self, node: ast.Call):
        func = node.func
        if isinstance(func, ast.Attribute):
            method = func.attr
            recv = func.value.id if isinstance(func.value, ast.Name) else None
            if method == "bind_parameters":
                self.findings.append(Finding("deprecated-bind-parameters", node.lineno,
                                             ".bind_parameters(...)"))
            if method == "get_counts":
                self.has_get_counts = True
            if method in ("measure", "measure_all", "measure_active"):
                self.has_measure = True
            dims = self.circuits.get(recv)
            if dims is not None:
                self._check_indices(method, node, dims)
        self.generic_visit(node)

    def _check_indices(self, method: str, node: ast.Call, dims):
        nq, nc = dims
        # qubit-index out of range
        if nq is not None:
            for pos in _qubit_arg_indices(method, len(node.args)):
                if pos < len(node.args):
                    self._flag_oob(node.args[pos], nq, "qubit-index-oob", node.lineno, method)
        # clbit checks for measure
        if method == "measure":
            cl = node.args[1] if len(node.args) > 1 else None
            q = node.args[0] if node.args else None
            if cl is not None and nc is not None:
                self._flag_oob(cl, nc, "clbit-size-mismatch", node.lineno, method)
            # measure(qreg, creg): a smaller classical register than quantum register
            if (isinstance(q, ast.Name) and isinstance(cl, ast.Name)
                    and self.reg_kind.get(q.id) == "q" and self.reg_kind.get(cl.id) == "c"):
                qs, cs = self.reg_sizes.get(q.id), self.reg_sizes.get(cl.id)
                if qs is not None and cs is not None and cs < qs:
                    self.findings.append(Finding("clbit-size-mismatch", node.lineno,
                                                 f"measure: {qs} qubits into {cs} clbits"))

    def _flag_oob(self, arg, bound: int, category: str, line: int, method: str):
        # literal int
        v = _const_int(arg)
        if v is not None and v >= bound:
            self.findings.append(Finding(category, line, f"{method}: index {v} >= {bound}"))
            return
        # list/tuple of literal ints
        if isinstance(arg, (ast.List, ast.Tuple)):
            for el in arg.elts:
                v = _const_int(el)
                if v is not None and v >= bound:
                    self.findings.append(Finding(category, line, f"{method}: index {v} >= {bound}"))


def detect(source: str) -> list[Finding]:
    """Return de-duplicated static findings for *source*."""
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [Finding("syntax-error", e.lineno or 0, str(e))]
    an = _Analyzer(source)
    an.visit(tree)
    # missing-measurement: counts requested but no measurement anywhere
    if an.has_get_counts and not an.has_measure:
        # attach to the get_counts line if findable
        line = next((i + 1 for i, ln in enumerate(source.splitlines())
                     if "get_counts" in ln), 0)
        an.findings.append(Finding("missing-measurement", line, "get_counts without measurement"))
    # de-dup by (category, line)
    seen, out = set(), []
    for f in an.findings:
        key = (f.category, f.line)
        if key not in seen:
            seen.add(key)
            out.append(f)
    return out


if __name__ == "__main__":
    import json
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    bench = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                         "benchmark", "benchmark.json")))
    for inst in bench:
        fs = detect(inst["buggy"])
        cats = sorted({f.category for f in fs})
        print(f"{inst['id']:24s} truth={inst['category']:26s} found={cats}")
