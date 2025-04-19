import numpy as np
from typing import List, Tuple, Optional

class QDensity:
    def __init__(self, num_qubits: int, states: Optional[List[Tuple[float, np.ndarray]]] = None):
        """Create a *num_qubits*‑qubit register.

        Parameters
        ----------
        num_qubits
            Number of qubits in the register.
        states
            Optional list of ``(probability, state_vector)`` tuples such that
            ``sum(p) == 1``.  Each *state_vector* must have length
            ``2 ** num_qubits``.  When *None* the register is initialised in
            |0…0⟩ with probability‑1.
        """
        self.num_qubits = num_qubits
        self.dim = 1 << num_qubits  # 2 ** num_qubits

        if states is None:
            # Start in the computational |0…0⟩ state.
            psi = np.zeros(self.dim, dtype=np.complex64)
            psi[0] = 1.0
            self.density_matrix = np.outer(psi, psi.conj())
            return

        self.density_matrix = np.zeros((self.dim, self.dim), dtype=np.complex64)

        prob_sum = 0.0
        for p, vec in states:
            if vec.shape != (self.dim,):
                raise ValueError("State‑vector has wrong length for this register.")
            self.density_matrix += p * np.outer(vec, vec.conj())
            prob_sum += p

        if not np.isclose(prob_sum, 1.0):
            raise ValueError(f"Probabilities must sum to 1 (got {prob_sum}).")

    def apply_gate(self, gate: np.ndarray, target: int) -> "QDensity":
        """Apply *gate* to the qubit(s) starting at *target*.

        The density matrix is transformed as ``ρ ↦ U ρ U†``.
        """
        self.__check_apply_gate_inputs(gate, target)
        gate_size = int(np.log2(gate.shape[0]))

        u_full = self.__add_right_identity_gates(gate, target)
        u_full = self.__add_left_identity_gates(u_full, self.num_qubits - target - gate_size)

        self.density_matrix = u_full @ self.density_matrix @ u_full.conj().T
        return self

    def get_density_matrix(self) -> np.ndarray:
        return self.density_matrix

    @staticmethod
    def __add_left_identity_gates(gate: np.ndarray, qtty: int) -> np.ndarray:
        for _ in range(qtty):
            gate = np.kron(np.eye(2, dtype=np.complex64), gate)
        return gate

    @staticmethod
    def __add_right_identity_gates(gate: np.ndarray, qtty: int) -> np.ndarray:
        for _ in range(qtty):
            gate = np.kron(gate, np.eye(2, dtype=np.complex64))
        return gate

    def __check_apply_gate_inputs(self, gate: np.ndarray, target: int) -> None:
        if gate.shape[0] != gate.shape[1]:
            raise ValueError("Gate must be square.")

        gate_size = int(np.log2(gate.shape[0]))
        if 2 ** gate_size != gate.shape[0]:
            raise ValueError("Gate dimension must be a power of two.")

        if target > self.num_qubits - gate_size:
            raise ValueError(
                f"Target qubit {target} outside the register for a gate of size {gate_size} qubits."
            )