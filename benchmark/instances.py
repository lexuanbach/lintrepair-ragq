"""LintRepair-RAG-Q pilot benchmark.

A curated set of Qiskit programs, each reproducing a *documented* real-world
quantum-software defect category and paired with a human-fixed reference.
Categories are drawn from:

* the Qiskit 1.0 API removals/migration (``execute``, ``qiskit.Aer``,
  ``bind_parameters``) -- defects that broke real code in the wild;
* structural defects reported by static analysers such as LintQ
  (out-of-range qubit/clbit indices, missing measurement, idle qubits); and
* silent-semantic defects in the Bugs4Q / Smelly-Eight tradition
  (wrong/forgotten gate, wrong rotation angle, swapped control/target,
  little-endian bit-order misreads).

Each entry is a dict:
  id, category, kind ("crash"|"semantic"|"smell"|"clean"),
  bug_line_substr (location label for detection scoring; "" for clean),
  buggy (source), fixed (source), note (provenance).

Every program binds module-level ``RESULT`` (counts dict or Statevector) on a
successful run.  Crash bugs raise before binding RESULT; semantic/smell bugs
bind a RESULT that differs from the fixed reference.  This is verified by
``build_benchmark.py`` -- nothing here is assumed, everything is executed.
"""

# Shared, correct Qiskit 2.x sampling boilerplate used by many fixed programs.
RUN = (
    "from qiskit import QuantumCircuit, transpile\n"
    "from qiskit_aer import AerSimulator\n"
)

INSTANCES = []


def add(**kw):
    INSTANCES.append(kw)


# ---------------------------------------------------------------------------
# Category A: deprecated-execute-api  (Qiskit 1.0 removed top-level execute())
# ---------------------------------------------------------------------------
add(
    id="A1-execute-bell",
    category="deprecated-execute-api",
    kind="crash",
    bug_line_substr="from qiskit import execute",
    note="Qiskit>=1.0 removed qiskit.execute; real migration defect.",
    buggy="""\
from qiskit import QuantumCircuit, execute, Aer
qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])
backend = Aer.get_backend('qasm_simulator')
job = execute(qc, backend, shots=8192, seed_simulator=1234)
RESULT = job.result().get_counts()
""",
    fixed="""\
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])
sim = AerSimulator(seed_simulator=1234)
RESULT = sim.run(transpile(qc, sim), shots=8192).result().get_counts()
""",
)
add(
    id="A2-execute-ghz",
    category="deprecated-execute-api",
    kind="crash",
    bug_line_substr="execute(",
    note="Qiskit 1.0 execute() removal on a 3-qubit GHZ preparation.",
    buggy="""\
import qiskit
from qiskit import QuantumCircuit
qc = QuantumCircuit(3, 3)
qc.h(0)
qc.cx(0, 1)
qc.cx(1, 2)
qc.measure(range(3), range(3))
backend = qiskit.Aer.get_backend('qasm_simulator')
RESULT = qiskit.execute(qc, backend, shots=8192, seed_simulator=7).result().get_counts()
""",
    fixed="""\
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
qc = QuantumCircuit(3, 3)
qc.h(0)
qc.cx(0, 1)
qc.cx(1, 2)
qc.measure(range(3), range(3))
sim = AerSimulator(seed_simulator=7)
RESULT = sim.run(transpile(qc, sim), shots=8192).result().get_counts()
""",
)
add(
    id="A3-execute-superposition",
    category="deprecated-execute-api",
    kind="crash",
    bug_line_substr="execute(",
    note="execute() removal on a single-qubit superposition.",
    buggy="""\
from qiskit import QuantumCircuit, execute
from qiskit import Aer
qc = QuantumCircuit(1, 1)
qc.h(0)
qc.measure(0, 0)
RESULT = execute(qc, Aer.get_backend('qasm_simulator'), shots=8192, seed_simulator=3).result().get_counts()
""",
    fixed="""\
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
qc = QuantumCircuit(1, 1)
qc.h(0)
qc.measure(0, 0)
sim = AerSimulator(seed_simulator=3)
RESULT = sim.run(transpile(qc, sim), shots=8192).result().get_counts()
""",
)
add(
    id="A4-execute-2bell",
    category="deprecated-execute-api",
    kind="crash",
    bug_line_substr="execute(",
    note="execute() removal on two independent Bell pairs.",
    buggy="""\
from qiskit import QuantumCircuit, execute, Aer
qc = QuantumCircuit(4, 4)
qc.h(0); qc.cx(0, 1)
qc.h(2); qc.cx(2, 3)
qc.measure(range(4), range(4))
RESULT = execute(qc, Aer.get_backend('aer_simulator'), shots=8192, seed_simulator=11).result().get_counts()
""",
    fixed="""\
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
qc = QuantumCircuit(4, 4)
qc.h(0); qc.cx(0, 1)
qc.h(2); qc.cx(2, 3)
qc.measure(range(4), range(4))
sim = AerSimulator(seed_simulator=11)
RESULT = sim.run(transpile(qc, sim), shots=8192).result().get_counts()
""",
)

# ---------------------------------------------------------------------------
# Category B: removed-aer-import  (qiskit.Aer moved to qiskit_aer in 1.0)
# ---------------------------------------------------------------------------
add(
    id="B1-aer-import",
    category="removed-aer-import",
    kind="crash",
    bug_line_substr="from qiskit import Aer",
    note="qiskit.Aer removed in 1.0; must import from qiskit_aer.",
    buggy="""\
from qiskit import QuantumCircuit, transpile
from qiskit import Aer
qc = QuantumCircuit(2, 2)
qc.h(0); qc.cx(0, 1)
qc.measure([0, 1], [0, 1])
sim = Aer.get_backend('aer_simulator')
RESULT = sim.run(transpile(qc, sim), shots=8192, seed_simulator=1234).result().get_counts()
""",
    fixed="""\
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
qc = QuantumCircuit(2, 2)
qc.h(0); qc.cx(0, 1)
qc.measure([0, 1], [0, 1])
sim = AerSimulator(seed_simulator=1234)
RESULT = sim.run(transpile(qc, sim), shots=8192).result().get_counts()
""",
)
add(
    id="B2-aer-qualified",
    category="removed-aer-import",
    kind="crash",
    bug_line_substr="qiskit.Aer",
    note="Qualified qiskit.Aer access removed in 1.0.",
    buggy="""\
import qiskit
from qiskit import QuantumCircuit, transpile
qc = QuantumCircuit(3, 3)
qc.h(range(3))
qc.measure(range(3), range(3))
sim = qiskit.Aer.get_backend('aer_simulator')
RESULT = sim.run(transpile(qc, sim), shots=8192, seed_simulator=5).result().get_counts()
""",
    fixed="""\
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
qc = QuantumCircuit(3, 3)
qc.h(range(3))
qc.measure(range(3), range(3))
sim = AerSimulator(seed_simulator=5)
RESULT = sim.run(transpile(qc, sim), shots=8192).result().get_counts()
""",
)
add(
    id="B3-aer-statevector",
    category="removed-aer-import",
    kind="crash",
    bug_line_substr="from qiskit import Aer",
    note="qiskit.Aer used for a statevector backend; removed in 1.0.",
    buggy="""\
from qiskit import QuantumCircuit, transpile
from qiskit import Aer
qc = QuantumCircuit(2)
qc.h(0); qc.cx(0, 1)
qc.save_statevector()
sim = Aer.get_backend('aer_simulator_statevector')
RESULT = sim.run(transpile(qc, sim)).result().get_statevector()
""",
    fixed="""\
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
qc = QuantumCircuit(2)
qc.h(0); qc.cx(0, 1)
RESULT = Statevector(qc)
""",
)
add(
    id="B4-aer-import-w",
    category="removed-aer-import",
    kind="crash",
    bug_line_substr="from qiskit import Aer",
    note="qiskit.Aer on a W-state-like circuit.",
    buggy="""\
from qiskit import QuantumCircuit, transpile
from qiskit import Aer
qc = QuantumCircuit(2, 2)
qc.ry(1.9106332362490186, 0)
qc.ch(0, 1)
qc.cx(1, 0)
qc.x(0)
qc.measure([0, 1], [0, 1])
sim = Aer.get_backend('aer_simulator')
RESULT = sim.run(transpile(qc, sim), shots=8192, seed_simulator=2).result().get_counts()
""",
    fixed="""\
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
qc = QuantumCircuit(2, 2)
qc.ry(1.9106332362490186, 0)
qc.ch(0, 1)
qc.cx(1, 0)
qc.x(0)
qc.measure([0, 1], [0, 1])
sim = AerSimulator(seed_simulator=2)
RESULT = sim.run(transpile(qc, sim), shots=8192).result().get_counts()
""",
)

# ---------------------------------------------------------------------------
# Category C: qubit-index-oob  (gate applied to a non-existent qubit)
# ---------------------------------------------------------------------------
add(
    id="C1-cx-oob",
    category="qubit-index-oob",
    kind="crash",
    bug_line_substr="qc.cx(0, 2)",
    note="CX targets qubit 2 in a 2-qubit register (CircuitError).",
    buggy="""\
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 2)
qc.measure([0, 1], [0, 1])
sim = AerSimulator(seed_simulator=1234)
RESULT = sim.run(transpile(qc, sim), shots=8192).result().get_counts()
""",
    fixed="""\
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])
sim = AerSimulator(seed_simulator=1234)
RESULT = sim.run(transpile(qc, sim), shots=8192).result().get_counts()
""",
)
add(
    id="C2-h-oob",
    category="qubit-index-oob",
    kind="crash",
    bug_line_substr="qc.h(3)",
    note="Hadamard on qubit 3 of a 3-qubit (indices 0-2) circuit.",
    buggy="""\
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
qc = QuantumCircuit(3, 3)
qc.h(3)
qc.cx(0, 1)
qc.measure(range(3), range(3))
sim = AerSimulator(seed_simulator=9)
RESULT = sim.run(transpile(qc, sim), shots=8192).result().get_counts()
""",
    fixed="""\
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
qc = QuantumCircuit(3, 3)
qc.h(0)
qc.cx(0, 1)
qc.measure(range(3), range(3))
sim = AerSimulator(seed_simulator=9)
RESULT = sim.run(transpile(qc, sim), shots=8192).result().get_counts()
""",
)
add(
    id="C3-x-oob",
    category="qubit-index-oob",
    kind="crash",
    bug_line_substr="qc.x(4)",
    note="X on qubit 4 of a 4-qubit (indices 0-3) circuit.",
    buggy="""\
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
qc = QuantumCircuit(4, 4)
qc.h(0); qc.cx(0, 1); qc.cx(1, 2); qc.cx(2, 3)
qc.x(4)
qc.measure(range(4), range(4))
sim = AerSimulator(seed_simulator=4)
RESULT = sim.run(transpile(qc, sim), shots=8192).result().get_counts()
""",
    fixed="""\
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
qc = QuantumCircuit(4, 4)
qc.h(0); qc.cx(0, 1); qc.cx(1, 2); qc.cx(2, 3)
qc.x(3)
qc.measure(range(4), range(4))
sim = AerSimulator(seed_simulator=4)
RESULT = sim.run(transpile(qc, sim), shots=8192).result().get_counts()
""",
)
add(
    id="C4-cz-oob",
    category="qubit-index-oob",
    kind="crash",
    bug_line_substr="qc.cz(1, 5)",
    note="CZ targets qubit 5 in a 3-qubit register.",
    buggy="""\
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
qc = QuantumCircuit(3, 3)
qc.h(1)
qc.cz(1, 5)
qc.measure(range(3), range(3))
sim = AerSimulator(seed_simulator=6)
RESULT = sim.run(transpile(qc, sim), shots=8192).result().get_counts()
""",
    fixed="""\
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
qc = QuantumCircuit(3, 3)
qc.h(1)
qc.cz(1, 2)
qc.measure(range(3), range(3))
sim = AerSimulator(seed_simulator=6)
RESULT = sim.run(transpile(qc, sim), shots=8192).result().get_counts()
""",
)

# ---------------------------------------------------------------------------
# Category D: clbit-size-mismatch  (classical register too small / bad index)
# ---------------------------------------------------------------------------
add(
    id="D1-creg-small",
    category="clbit-size-mismatch",
    kind="crash",
    bug_line_substr="QuantumCircuit(3, 2)",
    note="3 qubits measured into a 2-bit classical register.",
    buggy="""\
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
qc = QuantumCircuit(3, 2)
qc.h(0); qc.cx(0, 1); qc.cx(1, 2)
qc.measure([0, 1, 2], [0, 1, 2])
sim = AerSimulator(seed_simulator=1234)
RESULT = sim.run(transpile(qc, sim), shots=8192).result().get_counts()
""",
    fixed="""\
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
qc = QuantumCircuit(3, 3)
qc.h(0); qc.cx(0, 1); qc.cx(1, 2)
qc.measure([0, 1, 2], [0, 1, 2])
sim = AerSimulator(seed_simulator=1234)
RESULT = sim.run(transpile(qc, sim), shots=8192).result().get_counts()
""",
)
add(
    id="D2-clbit-index",
    category="clbit-size-mismatch",
    kind="crash",
    bug_line_substr="qc.measure(1, 2)",
    note="Measure into classical bit index 2 of a 2-bit register.",
    buggy="""\
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
qc = QuantumCircuit(2, 2)
qc.h(0); qc.cx(0, 1)
qc.measure(0, 0)
qc.measure(1, 2)
sim = AerSimulator(seed_simulator=1234)
RESULT = sim.run(transpile(qc, sim), shots=8192).result().get_counts()
""",
    fixed="""\
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
qc = QuantumCircuit(2, 2)
qc.h(0); qc.cx(0, 1)
qc.measure(0, 0)
qc.measure(1, 1)
sim = AerSimulator(seed_simulator=1234)
RESULT = sim.run(transpile(qc, sim), shots=8192).result().get_counts()
""",
)
add(
    id="D3-creg-obj-small",
    category="clbit-size-mismatch",
    kind="crash",
    bug_line_substr="ClassicalRegister(1",
    note="Explicit ClassicalRegister too small for the measurement.",
    buggy="""\
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
q = QuantumRegister(2, 'q')
c = ClassicalRegister(1, 'c')
qc = QuantumCircuit(q, c)
qc.h(q[0]); qc.cx(q[0], q[1])
qc.measure(q, c)
sim = AerSimulator(seed_simulator=1234)
RESULT = sim.run(transpile(qc, sim), shots=8192).result().get_counts()
""",
    fixed="""\
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
q = QuantumRegister(2, 'q')
c = ClassicalRegister(2, 'c')
qc = QuantumCircuit(q, c)
qc.h(q[0]); qc.cx(q[0], q[1])
qc.measure(q, c)
sim = AerSimulator(seed_simulator=1234)
RESULT = sim.run(transpile(qc, sim), shots=8192).result().get_counts()
""",
)
add(
    id="D4-no-creg",
    category="clbit-size-mismatch",
    kind="crash",
    bug_line_substr="QuantumCircuit(2)",
    note="No classical bits declared but measure() addresses clbit 0/1.",
    buggy="""\
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
qc = QuantumCircuit(2)
qc.h(0); qc.cx(0, 1)
qc.measure([0, 1], [0, 1])
sim = AerSimulator(seed_simulator=1234)
RESULT = sim.run(transpile(qc, sim), shots=8192).result().get_counts()
""",
    fixed="""\
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
qc = QuantumCircuit(2, 2)
qc.h(0); qc.cx(0, 1)
qc.measure([0, 1], [0, 1])
sim = AerSimulator(seed_simulator=1234)
RESULT = sim.run(transpile(qc, sim), shots=8192).result().get_counts()
""",
)

# ---------------------------------------------------------------------------
# Category E: missing-measurement  (counts requested with no measurement)
# ---------------------------------------------------------------------------
add(
    id="E1-no-measure",
    category="missing-measurement",
    kind="crash",
    bug_line_substr="get_counts",
    note="get_counts() on a circuit with no measurement (QiskitError).",
    buggy="""\
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
qc = QuantumCircuit(2, 2)
qc.h(0); qc.cx(0, 1)
sim = AerSimulator(seed_simulator=1234)
RESULT = sim.run(transpile(qc, sim), shots=8192).result().get_counts()
""",
    fixed="""\
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
qc = QuantumCircuit(2, 2)
qc.h(0); qc.cx(0, 1)
qc.measure([0, 1], [0, 1])
sim = AerSimulator(seed_simulator=1234)
RESULT = sim.run(transpile(qc, sim), shots=8192).result().get_counts()
""",
)
add(
    id="E2-no-measure-ghz",
    category="missing-measurement",
    kind="crash",
    bug_line_substr="get_counts",
    note="GHZ run for counts but never measured.",
    buggy="""\
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
qc = QuantumCircuit(3, 3)
qc.h(0); qc.cx(0, 1); qc.cx(1, 2)
sim = AerSimulator(seed_simulator=8)
RESULT = sim.run(transpile(qc, sim), shots=8192).result().get_counts()
""",
    fixed="""\
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
qc = QuantumCircuit(3, 3)
qc.h(0); qc.cx(0, 1); qc.cx(1, 2)
qc.measure(range(3), range(3))
sim = AerSimulator(seed_simulator=8)
RESULT = sim.run(transpile(qc, sim), shots=8192).result().get_counts()
""",
)
add(
    id="E3-partial-measure",
    category="missing-measurement",
    kind="semantic",
    bug_line_substr="qc.measure(0, 0)",
    note="Only one of two entangled qubits is measured; correlation lost.",
    buggy="""\
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
qc = QuantumCircuit(2, 2)
qc.h(0); qc.cx(0, 1)
qc.measure(0, 0)
sim = AerSimulator(seed_simulator=1234)
RESULT = sim.run(transpile(qc, sim), shots=8192).result().get_counts()
""",
    fixed="""\
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
qc = QuantumCircuit(2, 2)
qc.h(0); qc.cx(0, 1)
qc.measure([0, 1], [0, 1])
sim = AerSimulator(seed_simulator=1234)
RESULT = sim.run(transpile(qc, sim), shots=8192).result().get_counts()
""",
)
add(
    id="E4-no-measure-super",
    category="missing-measurement",
    kind="crash",
    bug_line_substr="get_counts",
    note="Superposition sampled for counts without measurement.",
    buggy="""\
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
qc = QuantumCircuit(1, 1)
qc.h(0)
sim = AerSimulator(seed_simulator=3)
RESULT = sim.run(transpile(qc, sim), shots=8192).result().get_counts()
""",
    fixed="""\
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
qc = QuantumCircuit(1, 1)
qc.h(0)
qc.measure(0, 0)
sim = AerSimulator(seed_simulator=3)
RESULT = sim.run(transpile(qc, sim), shots=8192).result().get_counts()
""",
)

# ---------------------------------------------------------------------------
# Category F: wrong-gate-or-param  (silent-semantic logic defects)
# ---------------------------------------------------------------------------
add(
    id="F1-cz-for-cx",
    category="wrong-gate-or-param",
    kind="semantic",
    bug_line_substr="qc.cz(0, 1)",
    note="CZ used where CX intended; Bell correlation lost (Bugs4Q-style).",
    buggy="""\
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cz(0, 1)
qc.measure([0, 1], [0, 1])
sim = AerSimulator(seed_simulator=1234)
RESULT = sim.run(transpile(qc, sim), shots=8192).result().get_counts()
""",
    fixed="""\
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])
sim = AerSimulator(seed_simulator=1234)
RESULT = sim.run(transpile(qc, sim), shots=8192).result().get_counts()
""",
)
add(
    id="F2-rx-degrees",
    category="wrong-gate-or-param",
    kind="semantic",
    bug_line_substr="qc.rx(90",
    note="Rotation angle given in degrees, not radians (classic unit bug).",
    buggy="""\
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
qc = QuantumCircuit(1, 1)
qc.rx(90, 0)
qc.measure(0, 0)
sim = AerSimulator(seed_simulator=1234)
RESULT = sim.run(transpile(qc, sim), shots=8192).result().get_counts()
""",
    fixed="""\
import math
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
qc = QuantumCircuit(1, 1)
qc.rx(math.pi / 2, 0)
qc.measure(0, 0)
sim = AerSimulator(seed_simulator=1234)
RESULT = sim.run(transpile(qc, sim), shots=8192).result().get_counts()
""",
)
add(
    id="F3-swapped-cx",
    category="wrong-gate-or-param",
    kind="semantic",
    bug_line_substr="qc.cx(1, 0)",
    note="Control/target swapped; control is |0> so CX is a no-op.",
    buggy="""\
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(1, 0)
qc.measure([0, 1], [0, 1])
sim = AerSimulator(seed_simulator=1234)
RESULT = sim.run(transpile(qc, sim), shots=8192).result().get_counts()
""",
    fixed="""\
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])
sim = AerSimulator(seed_simulator=1234)
RESULT = sim.run(transpile(qc, sim), shots=8192).result().get_counts()
""",
)
add(
    id="F4-missing-h",
    category="wrong-gate-or-param",
    kind="semantic",
    bug_line_substr="qc.cx(0, 1)",
    note="Forgotten Hadamard: no superposition, output collapses to 00.",
    buggy="""\
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
qc = QuantumCircuit(2, 2)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])
sim = AerSimulator(seed_simulator=1234)
RESULT = sim.run(transpile(qc, sim), shots=8192).result().get_counts()
""",
    fixed="""\
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])
sim = AerSimulator(seed_simulator=1234)
RESULT = sim.run(transpile(qc, sim), shots=8192).result().get_counts()
""",
)

# ---------------------------------------------------------------------------
# Category G: endianness-bit-order  (little-endian misread in post-processing)
# ---------------------------------------------------------------------------
add(
    id="G1-msb-read",
    category="endianness-bit-order",
    kind="semantic",
    bug_line_substr="key[0]",
    note="Qiskit counts keys are little-endian; key[0] is the highest qubit.",
    buggy="""\
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
qc = QuantumCircuit(3, 3)
qc.x(0)            # set qubit 0 to |1>
qc.measure(range(3), range(3))
sim = AerSimulator(seed_simulator=1234)
counts = sim.run(transpile(qc, sim), shots=8192).result().get_counts()
# probability that qubit 0 is 1
top = max(counts, key=counts.get)
RESULT = {'q0_is_one': int(top[0] == '1')}
""",
    fixed="""\
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
qc = QuantumCircuit(3, 3)
qc.x(0)
qc.measure(range(3), range(3))
sim = AerSimulator(seed_simulator=1234)
counts = sim.run(transpile(qc, sim), shots=8192).result().get_counts()
top = max(counts, key=counts.get)
RESULT = {'q0_is_one': int(top[-1] == '1')}
""",
)
add(
    id="G2-label-order",
    category="endianness-bit-order",
    kind="semantic",
    bug_line_substr="str(1) + str(0)",
    note="Expected outcome label built qubit-0-first; Qiskit prints qubit-0 last.",
    buggy="""\
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
qc = QuantumCircuit(2, 2)
qc.x(0)                 # q0=1, q1=0
qc.measure([0, 1], [0, 1])
sim = AerSimulator(seed_simulator=1234)
counts = sim.run(transpile(qc, sim), shots=8192).result().get_counts()
expected = str(1) + str(0)            # q0 then q1 -> '10' (wrong order)
RESULT = {'hits': counts.get(expected, 0)}
""",
    fixed="""\
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
qc = QuantumCircuit(2, 2)
qc.x(0)
qc.measure([0, 1], [0, 1])
sim = AerSimulator(seed_simulator=1234)
counts = sim.run(transpile(qc, sim), shots=8192).result().get_counts()
expected = str(0) + str(1)            # q1 then q0 -> '01' (Qiskit order)
RESULT = {'hits': counts.get(expected, 0)}
""",
)
add(
    id="G3-qubit2-read",
    category="endianness-bit-order",
    kind="semantic",
    bug_line_substr="top[2]",
    note="Reading qubit 2 by string index 2 (it is the lowest-index char).",
    buggy="""\
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
qc = QuantumCircuit(3, 3)
qc.x(2)            # set qubit 2 to |1>
qc.measure(range(3), range(3))
sim = AerSimulator(seed_simulator=1234)
counts = sim.run(transpile(qc, sim), shots=8192).result().get_counts()
top = max(counts, key=counts.get)
RESULT = {'q2_is_one': int(top[2] == '1')}
""",
    fixed="""\
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
qc = QuantumCircuit(3, 3)
qc.x(2)
qc.measure(range(3), range(3))
sim = AerSimulator(seed_simulator=1234)
counts = sim.run(transpile(qc, sim), shots=8192).result().get_counts()
top = max(counts, key=counts.get)
RESULT = {'q2_is_one': int(top[0] == '1')}
""",
)
add(
    id="G4-marked-state",
    category="endianness-bit-order",
    kind="semantic",
    bug_line_substr="format(marked",
    note="Grover-style marked-state label built in the wrong bit order.",
    buggy="""\
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
marked = 1  # mark basis state |001> (qubit 0 set)
qc = QuantumCircuit(3, 3)
qc.x(0)
qc.measure(range(3), range(3))
sim = AerSimulator(seed_simulator=1234)
counts = sim.run(transpile(qc, sim), shots=8192).result().get_counts()
label = format(marked, '03b')           # '001' (big-endian)
RESULT = {'hits': counts.get(label, 0)}  # key is '001' here, but...
""",
    fixed="""\
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
marked = 1
qc = QuantumCircuit(3, 3)
qc.x(0)
qc.measure(range(3), range(3))
sim = AerSimulator(seed_simulator=1234)
counts = sim.run(transpile(qc, sim), shots=8192).result().get_counts()
label = format(marked, '03b')[::-1]
RESULT = {'hits': counts.get(label, 0)}
""",
)

# ---------------------------------------------------------------------------
# Category H: deprecated-bind-parameters  (renamed to assign_parameters in 1.0)
# ---------------------------------------------------------------------------
add(
    id="H1-bind-params",
    category="deprecated-bind-parameters",
    kind="crash",
    bug_line_substr="bind_parameters",
    note="QuantumCircuit.bind_parameters removed in 1.0 -> assign_parameters.",
    buggy="""\
import math
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator
theta = Parameter('theta')
qc = QuantumCircuit(1, 1)
qc.rx(theta, 0)
qc.measure(0, 0)
bound = qc.bind_parameters({theta: math.pi})
sim = AerSimulator(seed_simulator=1234)
RESULT = sim.run(transpile(bound, sim), shots=8192).result().get_counts()
""",
    fixed="""\
import math
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator
theta = Parameter('theta')
qc = QuantumCircuit(1, 1)
qc.rx(theta, 0)
qc.measure(0, 0)
bound = qc.assign_parameters({theta: math.pi})
sim = AerSimulator(seed_simulator=1234)
RESULT = sim.run(transpile(bound, sim), shots=8192).result().get_counts()
""",
)
add(
    id="H2-bind-params-2",
    category="deprecated-bind-parameters",
    kind="crash",
    bug_line_substr="bind_parameters",
    note="bind_parameters on a two-parameter ansatz.",
    buggy="""\
import math
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator
a = Parameter('a'); b = Parameter('b')
qc = QuantumCircuit(2, 2)
qc.ry(a, 0); qc.ry(b, 1); qc.cx(0, 1)
qc.measure([0, 1], [0, 1])
bound = qc.bind_parameters({a: math.pi, b: 0.0})
sim = AerSimulator(seed_simulator=1234)
RESULT = sim.run(transpile(bound, sim), shots=8192).result().get_counts()
""",
    fixed="""\
import math
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator
a = Parameter('a'); b = Parameter('b')
qc = QuantumCircuit(2, 2)
qc.ry(a, 0); qc.ry(b, 1); qc.cx(0, 1)
qc.measure([0, 1], [0, 1])
bound = qc.assign_parameters({a: math.pi, b: 0.0})
sim = AerSimulator(seed_simulator=1234)
RESULT = sim.run(transpile(bound, sim), shots=8192).result().get_counts()
""",
)
add(
    id="H3-bind-list",
    category="deprecated-bind-parameters",
    kind="crash",
    bug_line_substr="bind_parameters",
    note="bind_parameters with a positional list of values.",
    buggy="""\
import math
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator
t = Parameter('t')
qc = QuantumCircuit(1, 1)
qc.h(0); qc.rz(t, 0); qc.h(0)
qc.measure(0, 0)
bound = qc.bind_parameters([math.pi])
sim = AerSimulator(seed_simulator=1234)
RESULT = sim.run(transpile(bound, sim), shots=8192).result().get_counts()
""",
    fixed="""\
import math
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator
t = Parameter('t')
qc = QuantumCircuit(1, 1)
qc.h(0); qc.rz(t, 0); qc.h(0)
qc.measure(0, 0)
bound = qc.assign_parameters([math.pi])
sim = AerSimulator(seed_simulator=1234)
RESULT = sim.run(transpile(bound, sim), shots=8192).result().get_counts()
""",
)
add(
    id="H4-bind-3q",
    category="deprecated-bind-parameters",
    kind="crash",
    bug_line_substr="bind_parameters",
    note="bind_parameters on a 3-qubit parameterised circuit.",
    buggy="""\
import math
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator
p = Parameter('p')
qc = QuantumCircuit(3, 3)
qc.h(0); qc.crx(p, 0, 1); qc.cx(1, 2)
qc.measure(range(3), range(3))
bound = qc.bind_parameters({p: math.pi / 2})
sim = AerSimulator(seed_simulator=1234)
RESULT = sim.run(transpile(bound, sim), shots=8192).result().get_counts()
""",
    fixed="""\
import math
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator
p = Parameter('p')
qc = QuantumCircuit(3, 3)
qc.h(0); qc.crx(p, 0, 1); qc.cx(1, 2)
qc.measure(range(3), range(3))
bound = qc.assign_parameters({p: math.pi / 2})
sim = AerSimulator(seed_simulator=1234)
RESULT = sim.run(transpile(bound, sim), shots=8192).result().get_counts()
""",
)

# ---------------------------------------------------------------------------
# Clean programs (ground-truth negatives for precision / false-positive rate)
# fixed == buggy; category "clean".
# ---------------------------------------------------------------------------
_CLEAN = [
    ("K1-bell", """\
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
qc = QuantumCircuit(2, 2)
qc.h(0); qc.cx(0, 1)
qc.measure([0, 1], [0, 1])
sim = AerSimulator(seed_simulator=1234)
RESULT = sim.run(transpile(qc, sim), shots=8192).result().get_counts()
"""),
    ("K2-ghz", """\
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
qc = QuantumCircuit(3, 3)
qc.h(0); qc.cx(0, 1); qc.cx(1, 2)
qc.measure(range(3), range(3))
sim = AerSimulator(seed_simulator=8)
RESULT = sim.run(transpile(qc, sim), shots=8192).result().get_counts()
"""),
    ("K3-statevector", """\
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
qc = QuantumCircuit(2)
qc.h(0); qc.cx(0, 1)
RESULT = Statevector(qc)
"""),
    ("K4-measure-all", """\
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
qc = QuantumCircuit(2)
qc.h(0); qc.cx(0, 1)
qc.measure_all()
sim = AerSimulator(seed_simulator=1234)
RESULT = sim.run(transpile(qc, sim), shots=8192).result().get_counts()
"""),
    ("K5-param", """\
import math
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator
theta = Parameter('theta')
qc = QuantumCircuit(1, 1)
qc.rx(theta, 0)
qc.measure(0, 0)
bound = qc.assign_parameters({theta: math.pi})
sim = AerSimulator(seed_simulator=1234)
RESULT = sim.run(transpile(bound, sim), shots=8192).result().get_counts()
"""),
    ("K6-endian-correct", """\
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
qc = QuantumCircuit(3, 3)
qc.x(0)
qc.measure(range(3), range(3))
sim = AerSimulator(seed_simulator=1234)
counts = sim.run(transpile(qc, sim), shots=8192).result().get_counts()
top = max(counts, key=counts.get)
RESULT = {'decoded': int(top[::-1], 2)}
"""),
    ("K7-rotation", """\
import math
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
qc = QuantumCircuit(1, 1)
qc.rx(math.pi / 2, 0)
qc.measure(0, 0)
sim = AerSimulator(seed_simulator=1234)
RESULT = sim.run(transpile(qc, sim), shots=8192).result().get_counts()
"""),
    ("K8-3q-chain", """\
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
qc = QuantumCircuit(3, 3)
qc.h(0); qc.cx(0, 1); qc.cx(1, 2)
qc.x(2)
qc.measure(range(3), range(3))
sim = AerSimulator(seed_simulator=4)
RESULT = sim.run(transpile(qc, sim), shots=8192).result().get_counts()
"""),
]
for _id, _src in _CLEAN:
    add(id=_id, category="clean", kind="clean", bug_line_substr="",
        note="Correct program; ground-truth negative.", buggy=_src, fixed=_src)


if __name__ == "__main__":
    from collections import Counter
    c = Counter(i["category"] for i in INSTANCES)
    print(f"{len(INSTANCES)} instances across {len(c)} categories")
    for k, v in sorted(c.items()):
        print(f"  {k:28s} {v}")
