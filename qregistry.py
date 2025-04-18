import numpy as np
import scipy as sp
import gates

# noinspection DuplicatedCode
# noinspection DuplicateLiteral
# noinspection PyMethodMayBeStatic
# noinspection SpellCheckingInspection
class QRegistry:
    def __init__(self, num_qubits):
        """
        Initializes the register to the specified number of qbits
        :param num_qubits: number of qbits
        """
        arr = np.zeros((2 ** num_qubits, 1), dtype=np.complex64)
        arr[0] = [1 + 0j]
        self.state = arr
        self.num_qubits = num_qubits
        self.rng = np.random.default_rng()

    def get_state(self):
        return self.state

    def apply_gate(self, gate, target):
        """
        Receives a gate, and apply it to the target qbit. If the gate spans more than one qbit, it is assumed
        the target qbit is the first qbit to apply the gate to. Per example, having a 4qbit register, and applying
        a CNOT (2 qbits gate) in target qbit 2 (3rd qbit), results in the CNOT being applied to qbits 2 and 3.
        :param gate:
        :param target:
        :return: the modified registry
        """
        self.__check_apply_gate_inputs(gate, target)
        gate_size = int(np.log2(gate.shape[0]))
        full_gate = self.__add_right_identity_gates(gate, target)
        full_gate = self.__add_left_identity_gates(full_gate, self.num_qubits - target - 1 - (gate_size - 1))
        self.state = full_gate.dot(self.state)
        return self

    def apply_gate_sparse(self, gate, target):
        """
        Same as apply_gate, but using scipy sparse matrices
        :param gate:
        :param target:
        :return: the modified registry
        """
        self.__check_apply_gate_inputs(gate, target)
        gate_size = int(np.log2(gate.shape[0]))
        gate = sp.sparse.csr_array(gate)
        full_gate = self.__add_right_identity_gates_sparse(gate, target)
        full_gate = self.__add_left_identity_gates_sparse(full_gate, self.num_qubits - target - 1 - (gate_size - 1))
        self.state = full_gate.dot(self.state)
        return self

    def value_prob(self, value):
        """
        Probability of obtaining a specific value after reading the registry.
        :param value: A positive number that must be equal or smaller than 2^n. Being "n" the number of qbits
        :return: the probability of getting the value. As a number between 0 and 1
        """
        if value >= self.state.size:
            raise ValueError("Value is outside the register. Value: " + str(value) + "; max possible value: " + str(
                self.state.size - 1))
        if value < 0:
            raise ValueError("Value must be positive. Value: " + str(value))
        return (self.state[value] * np.conjugate(self.state[value])).real

    def qbit_prob(self, target):
        """
        Probability a specific qbit has of being one
        :param target:
        :return: The probability a specific target qbit has of being one
        """
        if target > self.num_qubits - 1:
            raise ValueError("Target qbit is outside the register. Target qbit: " + str(target))
        is_one = True
        period = 2 ** target
        total_prob = 0
        for i in range(0, self.state.size):
            if i % period == 0:
                is_one = not is_one
            if is_one:
                total_prob += self.value_prob(i)
        total_prob = self.__correct_probability_value(total_prob)
        return total_prob

    def collapse(self, target, value, prob_one=None):
        """
        Forces the target qbit to collapse to a value, 0 or 1. It is the equivalent of making a measure on a qbit,
        and getting the indicated value. The probability of measuring a value=1 in the target qbit might have been
        calculated already, in which case the caller might provide a value for prob_one argument

        Per example a 2qbit register just initialized has a 100% probability of measuring 0 in both qbits,
        using this method with value=1 on both qbits, will convert the registry to value 3 (that is, value 1 in both
        qbits).
        :param target: the qbit that will be forced to collapse to a value
        :param value: either 0 or 1, the value to which the qbit will collaps
        :param prob_one: the probability the target qbit has of being 1
        :return: the modified registry
        """
        self.__check_collapse_inputs(target)
        is_one = True
        period = 2 ** target
        for i in range(0, self.state.size):
            if i % period == 0:
                is_one = not is_one
            if (is_one and value == 0) or (not is_one and value == 1):
                self.state[i] = 0

        # Calculates the norm of the state, and then divide the whole state vector by it to normalize it
        # If the "prob_one" parameter is provided, then the norm can be easily calculated from it.
        if prob_one is None:
            norm = np.linalg.norm(self.state)
        elif value == 1:
            norm = prob_one ** (1 / 2)
        else:
            norm = (1 - prob_one) ** (1 / 2)
        for i in range(self.state.size):
            self.state[i] /= norm
        return self

    def get_density_matrix(self):
        """
        :return: The density matrix. This simulator does not simulate errors caused by external
        factors, so the density matrix is just the dot product between the state and its transposed conjugated
        """
        return np.dot(self.state, self.state.conj().T)

    def get_bloch_coords(self, deg=False):
        """
        Calculates the bloch coordinates for a 1qbit register.
        :param deg: indicates whether to return the angles in radians (default) or in degrees
        :return: theta and phi angles
        """
        if self.state.size > 2:
            raise ValueError("Cannot return bloch coords for systems of more than 1 qbit")
        theta = np.arccos(self.state[0]) * 2
        phi = np.angle(self.state[1]) - np.angle(self.state[0])
        if deg:
            return theta * 360 / (2 * np.pi), phi * 360 / (2 * np.pi)
        else:
            return theta, phi

    def measure(self, target, rng=None):
        """
        Measures the specified target qbit
        :param target:
        :param rng: Optional, the randum number generator to use. Must implement rng.random() method
        :return: the modified registry and the measured value
        """
        prob_one = self.qbit_prob(target)
        random_number = self.rng.random() if rng is None else rng.random()
        value = 1 if random_number < prob_one else 0
        self.collapse(target, value, prob_one)
        return self, value

    def __check_apply_gate_inputs(self, gate, target):
        gate_size = int(np.log2(gate.shape[0]))
        if gate.shape[0] != gate.shape[1]:
            raise ValueError("Gate must be a square")
        if target > self.num_qubits - 1 - (gate_size - 1):
            raise ValueError(
                "Target qbit is outside the register. Target qbit: " + str(target) + "; gate size: " + str(gate_size))

    def __add_left_identity_gates(self, gate, qtty):
        for _ in range(qtty):
            gate = np.kron(gates.I(1), gate)
        return gate

    def __add_right_identity_gates(self, gate, qtty):
        for _ in range(qtty):
            gate = np.kron(gate, gates.I(1))
        return gate

    def __add_left_identity_gates_sparse(self, gate, qtty):
        for _ in range(qtty):
            gate = sp.sparse.kron(gates.I_sparse(1), gate, format="csr")
        return gate

    def __add_right_identity_gates_sparse(self, gate, qtty):
        for _ in range(qtty):
            gate = sp.sparse.kron(gate, gates.I_sparse(1), format="csr")
        return gate

    def __correct_probability_value(self, total_prob):
        max_error = 0.001
        if total_prob > 1 > total_prob - max_error:
            return 1
        elif total_prob > 1:
            raise ValueError("Error happend during the calculation, the probability is higher than 1")
        else:
            return total_prob

    def __check_collapse_inputs(self, target):
        if target > self.num_qubits - 1:
            raise ValueError("Target qbit is outside the register. Target qbit: " + str(target))
