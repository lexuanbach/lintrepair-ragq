"""Retrieval layer: a curated knowledge base of *general* quantum-software
defect patterns + a dependency-free TF-IDF retriever.

Design note on honesty: every KB document describes a defect *family* and its
remedy at the level a developer or the Qiskit migration guide would state it.
No document encodes the answer to a specific benchmark instance (no buggy line,
no per-instance fix), so retrieval cannot shortcut the task -- it only supplies
the same general knowledge a human would look up.  Distractor documents are
included so retrieval is non-trivial.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

KB_DOCS = [
    dict(id="kb-execute-removed", tag="Qiskit 1.0 migration",
         text="""Qiskit 1.0 removed the top-level execute() helper and qiskit.execute.
         Run circuits with AerSimulator().run(transpile(circuit, backend)).result(), or
         with the primitives SamplerV2/EstimatorV2. The old execute(circuit, backend,
         shots=...) call no longer exists and raises ImportError. Keywords: execute, run,
         transpile, deprecated, removed, migration, backend, shots."""),
    dict(id="kb-aer-moved", tag="Qiskit 1.0 migration",
         text="""The Aer simulator provider moved out of the qiskit namespace into the
         separate qiskit_aer package. Replace 'from qiskit import Aer' and 'qiskit.Aer.
         get_backend(...)' with 'from qiskit_aer import AerSimulator' and AerSimulator().
         Keywords: Aer, qiskit_aer, AerSimulator, get_backend, provider, import, removed."""),
    dict(id="kb-bind-renamed", tag="Qiskit 1.0 migration",
         text="""QuantumCircuit.bind_parameters was removed. Use assign_parameters, which
         accepts either a dict {parameter: value} or a positional list of values and returns
         a bound circuit. Keywords: bind_parameters, assign_parameters, Parameter, ansatz,
         parameterised, value binding."""),
    dict(id="kb-qubit-range", tag="circuit construction",
         text="""Gates and measurements address qubit indices in the half-open range
         [0, num_qubits). Using an index equal to or larger than num_qubits raises a
         CircuitError. Verify the QuantumCircuit was allocated with enough qubits for the
         largest index used by any gate. Keywords: qubit, index, out of range, CircuitError,
         num_qubits, register size."""),
    dict(id="kb-clbit-size", tag="circuit construction",
         text="""A measurement writes a qubit outcome into a classical bit; the classical
         register must be large enough. QuantumCircuit(n) allocates n qubits and zero
         classical bits. Use QuantumCircuit(n_qubits, n_clbits) or a ClassicalRegister sized
         to the number of qubits measured. A classical register smaller than the measured
         qubits raises a CircuitError. Keywords: classical bit, clbit, ClassicalRegister,
         measure, size mismatch, register."""),
    dict(id="kb-missing-measure", tag="testing / execution",
         text="""Sampling counts with AerSimulator and get_counts requires measurement
         instructions in the circuit. A circuit with no measure() or measure_all() produces
         no counts and get_counts raises a QiskitError. Add measure_all() or measure(qubits,
         clbits) before sampling. To inspect the exact state of a measurement-free circuit,
         use quantum_info.Statevector instead. Keywords: measurement, measure_all, get_counts,
         no counts, sampling, observation."""),
    dict(id="kb-endianness", tag="result interpretation",
         text="""Qiskit uses little-endian bit ordering. In a counts bitstring the rightmost
         character corresponds to qubit 0 and the leftmost to the highest-index qubit. When
         mapping classical post-processing back to qubit indices, index from the right or
         reverse the string with key[::-1]; when building an expected outcome label for a
         given integer or per-qubit assignment, reverse it to match Qiskit's order. Keywords:
         endianness, bit order, little-endian, reverse, qubit 0, counts key, label."""),
    dict(id="kb-entangle-measure", tag="result interpretation",
         text="""Observing the correlation of an entangled state (e.g. a Bell or GHZ state)
         requires measuring all entangled qubits. Measuring only a subset marginalises out
         the rest and destroys the observed correlation in the counts. Keywords: entanglement,
         Bell, GHZ, correlation, partial measurement, marginal."""),
    dict(id="kb-rotation-radians", tag="gate semantics",
         text="""Rotation gates rx, ry, rz, and p take their angle in radians, not degrees.
         A literal like rx(90, q) rotates by 90 radians and is almost never intended; use
         math.pi-based angles such as math.pi/2. Keywords: rotation, rx, ry, rz, angle,
         radians, degrees, math.pi."""),
    dict(id="kb-cx-vs-cz", tag="gate semantics",
         text="""cx (CNOT) flips its target conditioned on the control and produces
         computational-basis correlations; cz applies a controlled phase and does not flip
         in the computational basis. Using cz where cx was intended leaves a Hadamard
         superposition uncorrelated in the measured counts. Keywords: cx, cnot, cz, controlled
         phase, Bell state, entangling gate, wrong gate."""),
    dict(id="kb-control-target", tag="gate semantics",
         text="""In cx(control, target) the first argument is the control qubit and the
         second the target. Swapping them changes the semantics; if the intended control was
         placed in superposition but the call lists it as the target, the gate may act on a
         |0> control and become a no-op. Keywords: control, target, order, swapped, cx, cnot."""),
    dict(id="kb-transpile", tag="execution",
         text="""Before backend.run, transpile the circuit for the target backend so it uses
         the backend basis gates and respects the coupling map: transpile(circuit, backend).
         Keywords: transpile, backend, basis gates, coupling map, layout."""),
    # ---- distractors ----
    dict(id="kb-parameter-general", tag="parameterisation",
         text="""Parameterised circuits use circuit.Parameter and ParameterVector to declare
         symbolic angles, which are bound to concrete values before execution. Keywords:
         Parameter, ParameterVector, variational, ansatz, symbolic."""),
    dict(id="kb-statevector", tag="verification",
         text="""quantum_info.Statevector(circuit) returns the exact output state amplitudes
         of a measurement-free circuit and is useful for deterministic verification and
         fidelity comparison. Keywords: Statevector, amplitudes, fidelity, deterministic,
         quantum_info."""),
    dict(id="kb-seeding", tag="reproducibility",
         text="""For reproducible simulation set seed_simulator on AerSimulator and
         seed_transpiler in transpile. Fixed seeds make counts deterministic across runs.
         Keywords: seed, reproducible, deterministic, seed_simulator, seed_transpiler."""),
]

_WORD = re.compile(r"[a-zA-Z_][a-zA-Z_0-9]+")


def _tokens(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)]


@dataclass
class Retrieved:
    id: str
    tag: str
    text: str
    score: float


class TfidfRetriever:
    def __init__(self, docs=KB_DOCS):
        self.docs = docs
        self.doc_tokens = [_tokens(d["text"]) for d in docs]
        df = Counter()
        for toks in self.doc_tokens:
            for w in set(toks):
                df[w] += 1
        n = len(docs)
        self.idf = {w: math.log((1 + n) / (1 + c)) + 1.0 for w, c in df.items()}
        self.doc_vecs = [self._vec(toks) for toks in self.doc_tokens]

    def _vec(self, toks: list[str]) -> dict[str, float]:
        tf = Counter(toks)
        if not tf:
            return {}
        maxf = max(tf.values())
        return {w: (0.5 + 0.5 * f / maxf) * self.idf.get(w, math.log(len(self.docs)) + 1.0)
                for w, f in tf.items()}

    @staticmethod
    def _cos(a: dict, b: dict) -> float:
        if not a or not b:
            return 0.0
        common = set(a) & set(b)
        num = sum(a[w] * b[w] for w in common)
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        return num / (na * nb) if na and nb else 0.0

    def retrieve(self, query: str, k: int = 4) -> list[Retrieved]:
        qv = self._vec(_tokens(query))
        scored = [(self._cos(qv, dv), d) for dv, d in zip(self.doc_vecs, self.docs)]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [Retrieved(d["id"], d["tag"], " ".join(d["text"].split()), s)
                for s, d in scored[:k] if s > 0]


if __name__ == "__main__":
    import json
    import os
    bench = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                         "benchmark", "benchmark.json")))
    r = TfidfRetriever()
    for inst in bench[:8]:
        top = r.retrieve(inst["buggy"], k=3)
        print(f"{inst['id']:24s} {inst['category']:26s} -> {[t.id for t in top]}")
