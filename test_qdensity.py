import numpy as np
from unittest import TestCase

import gates
from qdensity import QDensity


tolerance = 0.0001 #0.01% for percentages, 0.0001 for scalars
class TestQDensity(TestCase):
    def test_constructor__initializes_matrix_to_0(self):
        qd = QDensity(1)
        expected = np.array([[1, 0], [0, 0]], dtype=np.complex128)
        assert np.allclose(qd.get_density_matrix(), expected)


    def test_constructor__when_statevectors_are_provided__then_initialize_accordingly(self):
        # |00⟩ with probability 0.6 and |11⟩ with probability 0.4
        psi0 = np.array([1, 0, 0, 0], dtype=np.complex128)
        psi1 = np.array([0, 0, 0, 1], dtype=np.complex128)
        states = [(0.6, psi0), (0.4, psi1)]

        qd = QDensity(2, states)
        print(qd.density_matrix)

        expected = 0.6 * np.outer(psi0, psi0.conj()) + 0.4 * np.outer(psi1, psi1.conj())
        assert np.allclose(qd.get_density_matrix(), expected)


    def test_apply_gate__should_work_for_simple_bell_state(self):
        qd = QDensity(2)
        qd.apply_gate(gates.h, 1)
        qd.apply_gate(gates.cnot, 0)

        expected = np.zeros((4, 4), dtype=np.complex128)
        expected[0, 0] = expected[3, 3] = 0.5
        expected[0, 3] = expected[3, 0] = 0.5

        assert np.allclose(qd.get_density_matrix(), expected, rtol=0, atol=tolerance), (
            f"Assertion failed, expected \n{expected}\n, but got \n{qd.get_density_matrix()}\n")
