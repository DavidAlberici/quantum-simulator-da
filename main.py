import numpy as np
import gates

class QRegistry:
    def __init__(self, num_qubits):
        arr = np.zeros((2 ** num_qubits, 1), dtype=np.complex64)
        arr[0] = [1 + 0j]
        self.state = arr
        self.num_qubits = num_qubits
        self.rng = np.random.default_rng()

    def get_state(self):
        return self.state

    def apply_gate(self, gate):
        if gate.shape[0] != gate.shape[1]:
            raise Exception("Gate must be a square (equal number of rows and columns)")
        if int(self.state.shape[0]) != int(gate.shape[0]):
            raise Exception("Gate must have the same number of rows as the state")
        self.state = gate.dot(self.state)
        return self

    # Does not work! Is the same as apply_gate
    def apply_unitary_gate(self, gate, target):
        if gate.shape[0] != gate.shape[1] or gate.shape[0] != 2:
            raise Exception("Gate must be 2x2")
        if target > self.num_qubits:
            raise Exception("Target qbit is outside the register")
        self.state = gate.dot(self.state)
        return self

    def apply_gate(self, gate, target):
        if gate.shape[0] != gate.shape[1]:
            raise Exception("Gate must be a square")
        gate_size = int(np.log2(gate.shape[0]))
        if target > self.num_qubits - 1 - (gate_size - 1):
            raise Exception(
                "Target qbit is outside the register. Target qbit: " + str(target) + "; gate size: " + str(gate_size))
        full_gate = self.__add_left_identity_gates(gate, target)
        full_gate = self.__add_right_identity_gates(full_gate, self.num_qubits - target - 1 - (gate_size - 1))
        self.state = full_gate.dot(self.state)
        return self

    @staticmethod
    def __add_left_identity_gates(gate, qtty):
        for _ in range(qtty):
            gate = np.kron(gates.Gates.I(1), gate)
        return gate

    @staticmethod
    def __add_right_identity_gates(gate, qtty):
        for _ in range(qtty):
            gate = np.kron(gate, gates.Gates.I(1))
        return gate

    # Probabilidad de obtener el valor value (1,2,3,4...) en todo el registro
    def value_prob(self, value):
        if value >= self.state.size:
            raise Exception("Value is outside the register. Value: " + str(value) + "; max possible value: " + str(
                self.state.size - 1))
        return (self.state[value] * np.conjugate(self.state[value])).real

    # Probabilidad de obtener el valor 1 en el qubit target
    def qbit_prob(self, target):
        if target > self.num_qubits - 1:
            raise Exception("Target qbit is outside the register. Target qbit: " + str(target))
        is_one = True
        period = 2 ** (target)
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
        if (total_prob > 1 and total_prob - max_error < 1):
            return 1
        elif (total_prob > 1):
            raise Exception("Error happend during the calculation, the probability is higher than 1")
        else:
            return total_prob

    # Forza el qubit target a colapsar al valor value (0,1)
    def collapse(self, target, value, prob_one=None):
        if (prob_one == None):
            return self.__collapse_without_prob(target, value)
        else:
            return self.__collapse_with_prob(target, value, prob_one)

    def __collapse_without_prob(self, target, value):
        if target > self.num_qubits - 1:
            raise Exception("Target qbit is outside the register. Target qbit: " + str(target))
        is_one = True
        period = 2 ** (target)
        for i in range(0, self.state.size):
            if i % period == 0:
                is_one = not is_one
            if ((is_one and value == 0)
                    or (not is_one and value == 1)):
                self.state[i] = 0
        self.state = self.state / np.linalg.norm(self.state)
        return self

    def __collapse_with_prob(self, target, value, prob_one):
        if target > self.num_qubits - 1:
            raise Exception("Target qbit is outside the register. Target qbit: " + str(target))
        is_one = True
        period = 2 ** (target)
        for i in range(0, self.state.size):
            if i % period == 0:
                is_one = not is_one
            if is_one and value == 0:
                self.state[i] = 0
            elif not is_one and value == 1:
                self.state[i] = 0

        if value == 1:
            self.state = self.state / prob_one ** (1 / 2)
        else:
            self.state = self.state / (1 - prob_one) ** (1 / 2)
        return self

    def measure(self, target):
        prob_one = self.qbit_prob(target)
        value = int(self.rng.random() < prob_one)
        self.collapse(target, value, value)
        return self, value

if __name__ == '__main__':
    print('PyCharm')

