import numpy as np
import gates
from q_registry_qbit import QRegistryPartialState



class QRegistryNew:
    def __init__(self, num_qubits):
        self.qbits = {}
        for i in range(num_qubits):
            self.qbits[i] = QRegistryPartialState([i])

    def get_qbit_num(self):
        return len(self.qbits)

    def apply_gate(self, gate, target=0):
        is_int = isinstance(target, int)
        is_int_list = isinstance(target, list) and all(isinstance(x, int) for x in target)
        if is_int:
            gate_size = int(np.log2(gate.shape[0]))
            new_target = []
            for i in range(gate_size):
                new_target.append(i+gate_size)
            return self.apply_gate_to_qbit_list(gate, new_target)
        elif is_int_list:
            return self.apply_gate_to_qbit_list(gate, target)
        else:
            raise ValueError("Target can only be a specific qbit index, or a list of qbits")

    def apply_gate_to_qbit_list(self, gate, target):
        self.__check_apply_gate_to_qbit_list_input(gate, target)
        partial_state = self.__find_partial_state_containing_target_qbits(target)
        partial_state.apply_gate(gate, target)
        return self

        #OLD
        # full_gate = self.__add_left_identity_gates(gate, target)
        # full_gate = self.__add_right_identity_gates(full_gate, self.get_qbit_num() - target - gate_size)
        # self.state = full_gate.dot(self.state)
        # return self

    def __check_apply_gate_to_qbit_list_input(self, gate, target):
        gate_size = int(np.log2(gate.shape[0]))
        if gate.shape[0] != gate.shape[1]:
            raise ValueError("Gate must be a square")
        if max(target) > self.get_qbit_num():
            raise ValueError("Target qbit is outside the register. Target qbit: " + str(max(target)))
        if len(target) > gate_size:
            raise ValueError(
                "Gate size does not correspond with target size. Gate size (# of qbits): " + str(gate_size) +
                "; target qbits: " + str(target)
            )

    def __find_partial_state_containing_target_qbits(self, target):
        state = None
        for qbit in target:
            if state is None:
                state = self.__get_qbit_state(qbit)
            elif qbit not in state.get_qbits():
                s2 = self.__get_qbit_state(qbit)
                state = self.__merge_states(state, s2)
        return state

    """
    The qubit state can be stored in 3 different ways, depending if it is grouped or not with other qbits. Bear in mind
    entangled qbits **must** be grouped together:
    1. The qbit is not entangled, so is directly stored in its index. Per example looking for qbit (index) 0, returns
    a state of size 2 as such [[0.707],[0.707]]
    2. The qbit is entangled, but is still stored in its index. Per example, the qbit 0 might be entangled with qbit 3 
    in the state [[0.707],[0],[0],[0.707]]
    3. The qbit is entangled and is not stored in its index. Per example, the qbit 3 might be entangled with qbit 2; in 
    this case the qbit 3 state is stored together with qbit 2. If you look for qbit 3, a reference to qbit 2 is returned
    """
    def __get_qbit_state(self, qbit_index):
        s = self.qbits.get(qbit_index)
        if isinstance(s, int):
            return self.__get_qbit_state(s)
        else:
            return s

    def __merge_states(self, state1, state2):
        if min(state1.get_qbits()) > min(state2.get_qbits()):
            s_temp = state2
            state1 = state2
            state2 = s_temp

        state1.merge_states(state2)

        min_qbit = min(state1.get_qbits())
        for i in state1.get_qbits():
            if i == min_qbit:
                self.qbits[i] = state1
            else:
                self.qbits[i] = min_qbit
        return state1

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
    def __add_identity_gate_from_the_left(gate, swaps):
        gate = np.kron(gates.Gates.I(1), gate)
        for _ in range(swaps):

            gate = np.kron(gates.Gates.I(1), gate)
        return gate
        pass


if __name__ == '__main__':
    a = [1,2,3]
    print(a[-1])

    # half = 1 / np.sqrt(2)
    # quarter = 1 / 2
    # quarter3 = np.sqrt(3) / 2
    # pos0 = np.array([[half], [half]])
    # pos1 = np.array([[1], [0]])
    # print(np.kron(pos1, pos0))
    # print("**************012")
    # pos02 = np.array([[1j], [2], [3j], [4]])
    # pos1 = np.array([[1], [10]])
    # qbits3 = np.kron(pos1, pos02)
    # print(qbits3)
    # print("**************")
    # swap2to3 = np.kron(gates.swap, gates.I(1))
    # print(np.dot(swap2to3, qbits3))
    # print("**************0132")
    # pos013 = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
    # pos2 = np.array([[1], [10]])
    # qbits3 = np.kron(pos2, pos013)
    # print(qbits3)
    #
    # print("**************0123")
    # swap2to3 = np.kron(gates.swap, gates.I(1))
    # swap2to3 = np.kron(swap2to3, gates.I(1))
    # print(np.dot(swap2to3, qbits3))
    # print("**************0213")
    # swap1to2 = np.kron(gates.swap, gates.I(1))
    # swap1to2 = np.kron(gates.I(1), swap1to2)
    # print(np.dot(swap1to2, qbits3))