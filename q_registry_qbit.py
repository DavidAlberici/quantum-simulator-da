import numpy as np
import gates

class QRegistryPartialState:
    def __init__(self, qbits):
        qbits_argument_is_numbers_array = isinstance(qbits, list) and all(isinstance(x, int) for x in qbits)
        if not qbits_argument_is_numbers_array:
            raise ValueError("qbits argument must be an array of integers")
        self.qbits = qbits
        self.state = self.__initialize_state_vector(qbits)

    def merge_states(self, other_state):
        self.__check_merge_states_input(other_state)
        new_qbits = sorted(self.qbits + other_state.get_qbits())
        new_size = (2 ** len(new_qbits))
        new_state = np.zeros((new_size, 1), dtype=np.complex64)
        for i in range(new_size):
            amp1_index = self.__extract_bits(i, self.qbits)
            amp1 = self.state[amp1_index][0]
            amp2_index = self.__extract_bits(i, other_state.get_qbits())
            amp2 = other_state.get_state()[amp2_index][0]
            new_state[i] = amp1 * amp2
        self.qbits = new_qbits
        self.state = new_state
        return self

    def apply_gate(self, gate, target):
        gate_size = int(np.log2(gate.shape[0]))
        target_index = self.qbits.index(target)
        self.__check_apply_gate_input(gate, gate_size, target_index)
        full_gate = self.__add_left_identity_gates(gate, target_index)
        full_gate = self.__add_right_identity_gates(full_gate, len(self.qbits) - target_index - 1 - (gate_size - 1))
        self.state = full_gate.dot(self.state)
        return self

    def get_state(self):
        return self.state

    def get_qbits(self):
        return self.qbits

    @staticmethod
    def __initialize_state_vector(qbits):
        arr = np.zeros((2 ** len(qbits), 1), dtype=np.complex64)
        arr[0] = [1 + 0j]
        return arr

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

    @staticmethod
    def __extract_bits(number, bit_positions):
        result = 0
        for i, pos in enumerate(sorted(bit_positions)):
            # Check if the bit is set in the original number
            if number & (1 << pos):
                # Set the bit in the result (at position i)
                result |= (1 << i)
        return result

    def __check_merge_states_input(self, other_state):
        if not isinstance(other_state, QRegistryPartialState):
            raise ValueError("The other state to be merged must be instance of QRegistryPartialState")
        if len(set(self.qbits + other_state.get_qbits())) > (len(self.qbits) + len(other_state.get_qbits())):
            raise ValueError("Both states have repeated qbits, and qbits should be unique")

    def __check_apply_gate_input(self, gate, gate_size, target_index):
        if gate.shape[0] != gate.shape[1]:
            raise ValueError("Gate must be a square")
        if target_index + gate_size > len(self.qbits):
            raise ValueError("Gate is too big, it requires " + str(gate_size) +
                             " qbits, but when applied in qbit index " + str(target_index) + " it only has "
                             + str(len(self.qbits) - target_index) + " qbits available")


if __name__ == '__main__':
    a = [1,2]
    b = [3,4]
    c = a + b
    print(c)