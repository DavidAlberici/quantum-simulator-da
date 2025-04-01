from unittest import TestCase

import numpy as np

import q_registry_qbit
import gates

class TestQRegistryPartialState(TestCase):
    def test_merge_states__should_return_correct_state_size(self):
        state1 = q_registry_qbit.QRegistryPartialState([0,2])
        state2 = q_registry_qbit.QRegistryPartialState([1,3])
        state1.merge_states(state2)
        self.assertEqual(state1.get_state().size, 16,
                         "The merge of 2 2-qbit states should return a state vector of size 16")

    def test_merge_states__should_merge_qbits_lists_correctly(self):
        get_state1 = lambda :q_registry_qbit.QRegistryPartialState([0,4])
        get_state2 = lambda :q_registry_qbit.QRegistryPartialState([2,3])
        get_state3 = lambda :q_registry_qbit.QRegistryPartialState([1,5])
        self.assertEqual(
            get_state1().merge_states(get_state2().merge_states(get_state3())).get_qbits(),
            get_state3().merge_states(get_state1().merge_states(get_state2())).get_qbits(),
            "Merged qbit list should be equal, regardless of merge order"
        )

    def test_merge_states__should_calculate_amplitudes_correctly(self):
        state1 = q_registry_qbit.QRegistryPartialState([0,2])
        state1.apply_gate(gates.h, 2)
        state2 = q_registry_qbit.QRegistryPartialState([1])
        state2.apply_gate(gates.h, 1)
        state1.merge_states(state2)
        # a hadamard was applied on qbits 1,2; this is the resulting vector state of qbits 0,1,2
        expected_state = np.array([[0.5], [0], [0.5], [0], [0.5], [0], [0.5], [0]])
        self.assertEqual(state1.get_state().all(), expected_state.all(),
                         "The merge should correctly multiply amplitudes")

    def test_apply_gate__should_apply_gate_to_correct_qbit(self):
        state1 = q_registry_qbit.QRegistryPartialState([0,2])
        state1.apply_gate(gates.h, 2)
        expected_state = np.array([[2**-0.5], [0], [2**-0.5], [0]])
        self.assertEqual(state1.get_state().all(), expected_state.all(),
                         "The merge of 2 2-qbit states should return a state vector of size 16")
