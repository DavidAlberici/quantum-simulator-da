import gates
import numpy as np
from qregistry import QRegistry
from qdensity import QDensity

if __name__ == '__main__':
    print("QRegistry examples:")
    # Use example: Initializing the registry
    qreg = QRegistry(2)

    # Use example: Getting the statevector
    print("Statevector:\n", qreg.get_state())

    # Use example: Applying a gate (Hadamard on qubit 0)
    qreg.apply_gate(gates.h, 0)
    print("After applying Hadamard to qubit 0:\n", qreg.get_state())

    # Use example: Creating a simple bell state (H on q0, then CNOT between q0 and q1)
    qreg = QRegistry(2)
    qreg.apply_gate(gates.h, 1)
    qreg.apply_gate(gates.cnot, 0)
    print("Bell state:\n", qreg.get_state())

    # Use example: Getting the probability of a value (value_prob)
    prob_val_0 = qreg.value_prob(0)
    print("Probability of measuring value 0:", prob_val_0)

    # Use example: Probability of a qbit being 1 (qbit_prob)
    prob_qbit1 = qreg.qbit_prob(1)
    print("Probability that qbit 1 is 1:", prob_qbit1)

    # Use example: Collapse a qbit to a value
    qreg.collapse(1, 1)
    print("After collapsing qbit 1 to 1:\n", qreg.get_state())

    # Use example: Get density matrix
    qreg = QRegistry(2)
    qreg.apply_gate(gates.h, 1)
    qreg.apply_gate(gates.cnot, 0)
    density = qreg.get_density_matrix()
    print("Density matrix:\n", density)

    # Use example: Get Bloch Sphere Coords (for a 1-qubit register)
    q1 = QRegistry(1)
    q1.apply_gate(gates.h, 0)
    theta, phi = q1.get_bloch_coords()
    print("Bloch coordinates (theta, phi) in radians:", theta, phi)

    # Use example: Measure
    qreg = QRegistry(2)
    qreg.apply_gate(gates.h, 0)
    qreg.apply_gate(gates.cnot, 0)
    rng = np.random.default_rng(1001)
    qreg, measured = qreg.measure(0, rng=rng)
    # For parallel measuring, just use the specific method (only works for 4 or more qbits)
    # qreg, measured = qreg.measure_parallel(0, rng=rng)
    print("Measured qbit 0:", measured)
    print("After measurement:\n", qreg.get_state())

    print("________________\nQDensity examples:")
    # Use example: Initializing the QDensity register (default state |00⟩)
    qd = QDensity(2)
    print("Initial density matrix:\n", qd.get_density_matrix())

    # Use example: Initializing the QDensity register with mixed states
    state1 = np.array([1, 0, 0, 0], dtype=np.complex64)  # |00⟩
    state2 = np.array([0, 1, 0, 0], dtype=np.complex64)  # |01⟩
    mixed_qd = QDensity(2, states=[
        (0.5, state1),
        (0.5, state2)
    ])
    print("Mixed state density matrix:\n", mixed_qd.get_density_matrix())

    # Use example: Applying a gate (Hadamard on qubit 0)
    qd = QDensity(2)
    qd.apply_gate(gates.h, 0)
    print("After applying Hadamard on qubit 0:\n", qd.get_density_matrix())

    # Use example: Applying a multi-qubit gate (CNOT on qubits 0 and 1)
    qd = QDensity(2)
    qd.apply_gate(gates.h, 1)
    qd.apply_gate(gates.cnot, 0)
    print("After applying H(1) then CNOT(0,1):\n", qd.get_density_matrix())

    # Use example: Getting the density matrix
    print("Current density matrix:\n", qd.get_density_matrix())