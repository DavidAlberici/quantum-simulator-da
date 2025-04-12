import numpy as np
import gates
import scipy as sp

# noinspection DuplicatedCode
# noinspection DuplicateLiteral
# noinspection PyMethodMayBeStatic
# noinspection SpellCheckingInspection
class QRegistry:
    def __init__(self, num_qubits):
        arr = np.zeros((2 ** num_qubits, 1), dtype=np.complex64)
        arr[0] = [1 + 0j]
        self.state = arr
        self.num_qubits = num_qubits
        self.rng = np.random.default_rng()

    def get_state(self):
        return self.state

    def apply_unitary_gate(self, gate, target):
        self.__check_apply_unitary_gate_inputs(gate, target)
        self.state = gate.dot(self.state)
        return self

    def apply_gate(self, gate, target):
        self.__check_apply_gate_inputs(gate, target)
        gate_size = int(np.log2(gate.shape[0]))
        full_gate = self.__add_left_identity_gates(gate, target)
        full_gate = self.__add_right_identity_gates(full_gate, self.num_qubits - target - 1 - (gate_size - 1))
        self.state = full_gate.dot(self.state)
        return self

    def apply_gate_sparse(self, gate, target):
        self.__check_apply_gate_inputs(gate, target)
        gate_size = int(np.log2(gate.shape[0]))
        gate = sp.sparse.csr_array(gate)
        full_gate = self.__add_left_identity_gates_sparse(gate, target)
        full_gate = self.__add_right_identity_gates_sparse(full_gate, self.num_qubits - target - 1 - (gate_size - 1))
        self.state = full_gate.dot(self.state)
        return self

    # Probabilidad de obtener el valor value (1,2,3,4...) en todo el registro
    def value_prob(self, value):
        if value >= self.state.size:
            raise ValueError("Value is outside the register. Value: " + str(value) + "; max possible value: " + str(
                self.state.size - 1))
        return (self.state[value] * np.conjugate(self.state[value])).real

    # Probabilidad de obtener el valor 1 en el qubit target
    def qbit_prob(self, target):
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
        total_prob = self.get_fixed_total_prob(total_prob)
        return total_prob

    def get_fixed_total_prob(self, total_prob):
        max_error = 0.001
        if total_prob > 1 > total_prob - max_error:
            return 1
        elif total_prob > 1:
            raise ValueError("Error happend during the calculation, the probability is higher than 1")
        else:
            return total_prob

    # Forza el qubit target a colapsar al valor value (0,1)
    def collapse(self, target, value, prob_one=None):
        if prob_one is None:
            return self.__collapse_without_prob(target, value)
        else:
            return self.__collapse_with_prob(target, value, prob_one)

    def get_density_matrix(self):
        return np.dot(self.state, self.state.conj().T)

    def get_bloch_coords(self, deg=False):
        if self.state.size > 2:
            raise ValueError("Cannot return bloch coords for systems of more than 1 qbit")
        theta = np.arccos(self.state[0]) * 2
        phi = np.angle(self.state[1]) - np.angle(self.state[0])
        if deg:
            return theta * 360 / (2 * np.pi), phi * 360 / (2 * np.pi)
        else:
            return theta, phi

    def measure(self, target):
        prob_one = self.qbit_prob(target)
        value = 1 if self.rng.random() < prob_one else 0
        self.collapse(target, value, value)
        return self, value

    def __check_apply_unitary_gate_inputs(self, gate, target):
        if gate.shape[0] != gate.shape[1] or gate.shape[0] != 2:
            raise ValueError("Gate must be 2x2")
        if target > self.num_qubits:
            raise ValueError("Target qbit is outside the register")

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

    def __collapse_without_prob(self, target, value):
        if target > self.num_qubits - 1:
            raise ValueError("Target qbit is outside the register. Target qbit: " + str(target))
        is_one = True
        period = 2 ** target
        for i in range(0, self.state.size):
            if i % period == 0:
                is_one = not is_one
            if (is_one and value == 0) or (not is_one and value == 1):
                self.state[i] = 0
        norm = np.linalg.norm(self.state)
        for i in range(self.state.size):
            self.state[i] /= norm
        return self

    def __collapse_with_prob(self, target, value, prob_one):
        if target > self.num_qubits - 1:
            raise ValueError("Target qbit is outside the register. Target qbit: " + str(target))
        is_one = True
        period = 2 ** target
        for i in range(0, self.state.size):
            if i % period == 0:
                is_one = not is_one
            if (is_one and value == 0) or (not is_one and value == 1):
                self.state[i] = 0

        amp = 0
        if value == 1:
            amp = prob_one ** (1 / 2)
        else:
            amp = (1 - prob_one) ** (1 / 2)
        for i in range(self.state.size):
            self.state[i] /= amp
        return self


if __name__ == '__main__':
    print('PyCharm')

    num_qbits = 10
    # Este Qreg es el "nuevo", con scipy
    qreg = QRegistry(num_qbits)
    qreg.apply_gate(gates.x, num_qbits - 1)
    print(qreg.get_state()[1])
    print(qreg.measure(0)[1])
    # 0.2s para 14qbits

    # Este Qreg usa np.array, sin scipy
    # qreg = QRegistry3103(num_qbits)
    # qreg.apply_gate(gates.x, num_qbits - 1)
    # print(qreg.get_state()[1])
    # print(qreg.measure(0))
    # #7.8s para 14qbits

    ## Usar multiprocessing para intentar paralelizar!

