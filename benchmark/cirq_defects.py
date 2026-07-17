"""Cirq-Defects: a small, calibrated Cirq silent-bug benchmark for the cross-SDK
transfer test. Each program binds RESULT (a counts dict, an int, or a list) and
carries an intent docstring, mirroring Q-Defects40. Bugs are SILENT-SEMANTIC and
deliberately in the difficulty sweet-spot: realistic enough that the bug is buried
and a ``no structural defects'' verdict is plausible (and, for the post-processing
/ endianness family, literally TRUE of the circuit), yet with clear intent so a
capable model can fix them unguided. No crash/API bugs (those are Qiskit-specific).
"""

# (id, family, correct_code, buggy_code). Both bind RESULT and run under cirq.
PROGRAMS = [
# ---- endianness / post-processing: the circuit is correct, the DECODING is wrong ----
("E1-decode-msb", "endianness-postproc",
 '''"""Prepare basis state with qubit 0 set (|100>) and DECODE the measured
bitstring as an integer with qubit 0 as the most-significant bit. Expect 4."""
import cirq
q=cirq.LineQubit.range(3)
c=cirq.Circuit([cirq.X(q[0]), cirq.measure(*q,key='m')])
bits=cirq.Simulator(seed=1).run(c,repetitions=1).measurements['m'][0]
RESULT=int("".join(str(int(b)) for b in bits),2)''',
 '''"""Prepare basis state with qubit 0 set (|100>) and DECODE the measured
bitstring as an integer with qubit 0 as the most-significant bit. Expect 4."""
import cirq
q=cirq.LineQubit.range(3)
c=cirq.Circuit([cirq.X(q[0]), cirq.measure(*q,key='m')])
bits=cirq.Simulator(seed=1).run(c,repetitions=1).measurements['m'][0]
RESULT=int("".join(str(int(b)) for b in bits[::-1]),2)'''),

("E2-counts-relabel", "endianness-postproc",
 '''"""Prepare |001> (qubit 2 set), sample, and return counts keyed by the
3-bit string with qubit 0 first. The only nonzero key should be "001"."""
import cirq
q=cirq.LineQubit.range(3)
c=cirq.Circuit([cirq.X(q[2]), cirq.measure(*q,key='m')])
rows=cirq.Simulator(seed=1).run(c,repetitions=200).measurements['m']
RESULT={}
for r in rows:
    k="".join(str(int(b)) for b in r); RESULT[k]=RESULT.get(k,0)+1''',
 '''"""Prepare |001> (qubit 2 set), sample, and return counts keyed by the
3-bit string with qubit 0 first. The only nonzero key should be "001"."""
import cirq
q=cirq.LineQubit.range(3)
c=cirq.Circuit([cirq.X(q[2]), cirq.measure(*q,key='m')])
rows=cirq.Simulator(seed=1).run(c,repetitions=200).measurements['m']
RESULT={}
for r in rows:
    k="".join(str(int(b)) for b in reversed(r)); RESULT[k]=RESULT.get(k,0)+1'''),

("E3-parity", "endianness-postproc",
 '''"""Prepare |100> (qubit 0 set) and return the integer value (qubit 0 = MSB).
Expect 4."""
import cirq
q=cirq.LineQubit.range(3)
c=cirq.Circuit([cirq.X(q[0]), cirq.measure(*q,key='m')])
b=cirq.Simulator(seed=1).run(c,repetitions=1).measurements['m'][0]
RESULT=sum(int(v)*(2**(len(b)-1-i)) for i,v in enumerate(b))''',
 '''"""Prepare |100> (qubit 0 set) and return the integer value (qubit 0 = MSB).
Expect 4."""
import cirq
q=cirq.LineQubit.range(3)
c=cirq.Circuit([cirq.X(q[0]), cirq.measure(*q,key='m')])
b=cirq.Simulator(seed=1).run(c,repetitions=1).measurements['m'][0]
RESULT=sum(int(v)*(2**i) for i,v in enumerate(b))'''),

# ---- wrong gate ----
("G1-cz-for-cx", "wrong-gate",
 '''"""Prepare a Bell state; return measurement counts. Only 00 and 11 appear."""
import cirq
q=cirq.LineQubit.range(2)
c=cirq.Circuit([cirq.H(q[0]),cirq.CNOT(q[0],q[1]),cirq.measure(*q,key='m')])
rows=cirq.Simulator(seed=1).run(c,repetitions=400).measurements['m']
RESULT={}
for r in rows:
    k="".join(str(int(b)) for b in r); RESULT[k]=RESULT.get(k,0)+1''',
 '''"""Prepare a Bell state; return measurement counts. Only 00 and 11 appear."""
import cirq
q=cirq.LineQubit.range(2)
c=cirq.Circuit([cirq.H(q[0]),cirq.CZ(q[0],q[1]),cirq.measure(*q,key='m')])
rows=cirq.Simulator(seed=1).run(c,repetitions=400).measurements['m']
RESULT={}
for r in rows:
    k="".join(str(int(b)) for b in r); RESULT[k]=RESULT.get(k,0)+1'''),

("G2-swapped-cnot", "wrong-gate",
 '''"""Set qubit 0 to |1>, then copy it onto qubit 1 with a CNOT controlled by
qubit 0. Return counts; expect only "11"."""
import cirq
q=cirq.LineQubit.range(2)
c=cirq.Circuit([cirq.X(q[0]),cirq.CNOT(q[0],q[1]),cirq.measure(*q,key='m')])
rows=cirq.Simulator(seed=1).run(c,repetitions=200).measurements['m']
RESULT={}
for r in rows:
    k="".join(str(int(b)) for b in r); RESULT[k]=RESULT.get(k,0)+1''',
 '''"""Set qubit 0 to |1>, then copy it onto qubit 1 with a CNOT controlled by
qubit 0. Return counts; expect only "11"."""
import cirq
q=cirq.LineQubit.range(2)
c=cirq.Circuit([cirq.X(q[0]),cirq.CNOT(q[1],q[0]),cirq.measure(*q,key='m')])
rows=cirq.Simulator(seed=1).run(c,repetitions=200).measurements['m']
RESULT={}
for r in rows:
    k="".join(str(int(b)) for b in r); RESULT[k]=RESULT.get(k,0)+1'''),

("G3-ch-for-cx", "wrong-gate",
 '''"""GHZ-like: H on q0 then CNOT(q0->q1) then CNOT(q1->q2); return counts.
Only "000" and "111" appear."""
import cirq
q=cirq.LineQubit.range(3)
c=cirq.Circuit([cirq.H(q[0]),cirq.CNOT(q[0],q[1]),cirq.CNOT(q[1],q[2]),cirq.measure(*q,key='m')])
rows=cirq.Simulator(seed=1).run(c,repetitions=400).measurements['m']
RESULT={}
for r in rows:
    k="".join(str(int(b)) for b in r); RESULT[k]=RESULT.get(k,0)+1''',
 '''"""GHZ-like: H on q0 then CNOT(q0->q1) then CNOT(q1->q2); return counts.
Only "000" and "111" appear."""
import cirq
q=cirq.LineQubit.range(3)
c=cirq.Circuit([cirq.H(q[0]),cirq.CNOT(q[0],q[1]),cirq.CZ(q[1],q[2]),cirq.measure(*q,key='m')])
rows=cirq.Simulator(seed=1).run(c,repetitions=400).measurements['m']
RESULT={}
for r in rows:
    k="".join(str(int(b)) for b in r); RESULT[k]=RESULT.get(k,0)+1'''),

# ---- wrong angle / parameter ----
("P1-degrees", "wrong-param",
 '''"""Rotate qubit 0 so that P(measuring 1) is exactly 0.25, using a Y-rotation
by pi/3 radians. Return counts over 400 shots."""
import cirq, numpy as np
q=cirq.LineQubit.range(1)
c=cirq.Circuit([cirq.ry(np.pi/3)(q[0]),cirq.measure(*q,key='m')])
rows=cirq.Simulator(seed=1).run(c,repetitions=400).measurements['m']
RESULT={}
for r in rows:
    k="".join(str(int(b)) for b in r); RESULT[k]=RESULT.get(k,0)+1''',
 '''"""Rotate qubit 0 so that P(measuring 1) is exactly 0.25, using a Y-rotation
by pi/3 radians. Return counts over 400 shots."""
import cirq, numpy as np
q=cirq.LineQubit.range(1)
c=cirq.Circuit([cirq.ry(60)(q[0]),cirq.measure(*q,key='m')])
rows=cirq.Simulator(seed=1).run(c,repetitions=400).measurements['m']
RESULT={}
for r in rows:
    k="".join(str(int(b)) for b in r); RESULT[k]=RESULT.get(k,0)+1'''),

("P2-half-angle", "wrong-param",
 '''"""Apply a Y-rotation putting qubit 0 into an EQUAL superposition
(P(0)=P(1)=0.5), i.e. Ry(pi/2). Return counts."""
import cirq, numpy as np
q=cirq.LineQubit.range(1)
c=cirq.Circuit([cirq.ry(np.pi/2)(q[0]),cirq.measure(*q,key='m')])
rows=cirq.Simulator(seed=1).run(c,repetitions=400).measurements['m']
RESULT={}
for r in rows:
    k="".join(str(int(b)) for b in r); RESULT[k]=RESULT.get(k,0)+1''',
 '''"""Apply a Y-rotation putting qubit 0 into an EQUAL superposition
(P(0)=P(1)=0.5), i.e. Ry(pi/2). Return counts."""
import cirq, numpy as np
q=cirq.LineQubit.range(1)
c=cirq.Circuit([cirq.ry(np.pi/4)(q[0]),cirq.measure(*q,key='m')])
rows=cirq.Simulator(seed=1).run(c,repetitions=400).measurements['m']
RESULT={}
for r in rows:
    k="".join(str(int(b)) for b in r); RESULT[k]=RESULT.get(k,0)+1'''),

# ---- wrong basis / phase ----
("B1-missing-s", "wrong-basis",
 '''"""Apply H, then S (phase), then H to qubit 0, then measure. This sequence
should yield a biased outcome (not 50/50). Return counts over 400 shots."""
import cirq
q=cirq.LineQubit.range(1)
c=cirq.Circuit([cirq.H(q[0]),cirq.S(q[0]),cirq.H(q[0]),cirq.measure(*q,key='m')])
rows=cirq.Simulator(seed=1).run(c,repetitions=400).measurements['m']
RESULT={}
for r in rows:
    k="".join(str(int(b)) for b in r); RESULT[k]=RESULT.get(k,0)+1''',
 '''"""Apply H, then S (phase), then H to qubit 0, then measure. This sequence
should yield a biased outcome (not 50/50). Return counts over 400 shots."""
import cirq
q=cirq.LineQubit.range(1)
c=cirq.Circuit([cirq.H(q[0]),cirq.H(q[0]),cirq.measure(*q,key='m')])
rows=cirq.Simulator(seed=1).run(c,repetitions=400).measurements['m']
RESULT={}
for r in rows:
    k="".join(str(int(b)) for b in r); RESULT[k]=RESULT.get(k,0)+1'''),

("B2-x-basis", "wrong-basis",
 '''"""Prepare |+> on qubit 0 (H|0>) and measure it in the X basis (apply H
before measuring) so the result is deterministically 0. Return counts."""
import cirq
q=cirq.LineQubit.range(1)
c=cirq.Circuit([cirq.H(q[0]),cirq.H(q[0]),cirq.measure(*q,key='m')])
rows=cirq.Simulator(seed=1).run(c,repetitions=200).measurements['m']
RESULT={}
for r in rows:
    k="".join(str(int(b)) for b in r); RESULT[k]=RESULT.get(k,0)+1''',
 '''"""Prepare |+> on qubit 0 (H|0>) and measure it in the X basis (apply H
before measuring) so the result is deterministically 0. Return counts."""
import cirq
q=cirq.LineQubit.range(1)
c=cirq.Circuit([cirq.H(q[0]),cirq.measure(*q,key='m')])
rows=cirq.Simulator(seed=1).run(c,repetitions=200).measurements['m']
RESULT={}
for r in rows:
    k="".join(str(int(b)) for b in r); RESULT[k]=RESULT.get(k,0)+1'''),
]
