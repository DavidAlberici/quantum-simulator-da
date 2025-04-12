import numpy as np
import gates

class QRegistryPartialState:
    def __init__(self, qbits):
        self.__check_constructor_input(qbits)
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
        self.__check_apply_gate_input(gate, target)
        modified_gate = self.__add_right_identy_gates(gate, target)
        gate_size = int(np.log2(gate.shape[0]))
        first_qbit_index = self.qbits.index(min(target))
        full_gate = self.__add_left_identity_gates(gate, first_qbit_index)
        full_gate = self.__add_right_identity_gates(full_gate, len(self.qbits) - first_qbit_index - gate_size)
        self.state = full_gate.dot(self.state)
        return self


    def get_state(self):
        return self.state

    def get_amplitudes_dictionary(self):
        dict = {}
        for i in range(self.state.size):
            dict[i] = np.round(float(self.state[i][0]), 4)
        return dict

    def print_probability(self, binary=False):
        for i in range(self.state.size):
            prob = np.absolute(self.state[i][0]) ** 2
            if binary:
                print(format(i,'0'+str(len(self.qbits))+'b'),": ", format(prob,".4f"))
            else:
                print(i,": ", format(prob,".4f"))

    def get_qbits(self):
        return self.qbits

    @staticmethod
    def __initialize_state_vector(qbits):
        arr = np.zeros((2 ** len(qbits), 1), dtype=np.complex64)
        arr[0] = [1 + 0j]
        return arr

    def __add_right_identity_gates(self, gate, target):
        number_of_gates = len(self.qbits) - len(target)
        for _ in range(number_of_gates):
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

    def __swap_qbit_left(self, qbit, swaps):
        ind = self.qbits.index(qbit)
        for i in range(swaps):
            pass


    def __check_merge_states_input(self, other_state):
        if not isinstance(other_state, QRegistryPartialState):
            raise ValueError("The other state to be merged must be instance of QRegistryPartialState")
        if len(set(self.qbits + other_state.get_qbits())) > (len(self.qbits) + len(other_state.get_qbits())):
            raise ValueError("Both states have repeated qbits, and qbits should be unique")

    def __check_apply_gate_input(self, gate, target):
        gate_size = int(np.log2(gate.shape[0]))
        if gate.shape[0] != gate.shape[1]:
            raise ValueError("Gate must be a square")
        if not self.__contains_target_qbits(target):
            raise ValueError("Unexpected error, QRegistryQbit can only apply a gate to qbits which it contains")
        if not gate_size == len(target):
            raise ValueError("Gate size does not match target qbits list size. Gate size: " + str(gate_size) +
                             ";  target qbits size: " + str(len(target)))

    def __contains_target_qbits(self, target):
        if len(target) > len(self.qbits):
            return False
        for qbit in target:
            try:
                self.qbits.index(qbit)
            except ValueError:
                return False
        return True

    def __check_constructor_input(self, qbits):
        qbits_argument_is_numbers_array = isinstance(qbits, list) and all(isinstance(x, int) for x in qbits)
        if not qbits_argument_is_numbers_array:
            raise ValueError("qbits argument must be an array of integers")


if __name__ == '__main__':
    state = QRegistryPartialState([0,1,2])
    state.apply_gate(gates.h, [0])
    state.apply_gate(gates.h, [2])
    state.apply_gate(gates.x, [2])
    state.print_probability(True)

    ## TODO usar esto para cambiar [0,1,2,3,4] a arr = [2,3,1,0,4]
    ## hacer un for para poblar un new_state, tal que new_state[permute_bits(i, arr)] = state[i]

    ## Hacer con
    def permute_bits(num, arr):
        result = 0
        for i in range(len(arr)):
            # Extract the bit from the original number at position `arr[i]`
            bit = (num >> arr[i]) & 1
            # Set this bit in the result at position `i`
            result |= (bit << i)
        return result