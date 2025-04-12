from unittest import TestCase
import qregistry
import gates
import numpy as np

tolerance = 0.0001 #0.01% for percentages, 0.0001 for scalars
class TestQRegistry(TestCase):
    def test_init__should_initialize_qregistry_to_0(self):
        q = qregistry.QRegistry(2)
        q2 = qregistry.QRegistry(7)
        assert q.get_state()[0] == 1, "The qregistry should be initialized to 0"
        assert q2.get_state()[0] == 1, "The qregistry should be initialized to 0"

    def test_apply_gate__should_work_for_simple_bell_state(self):
        q = self.__get_qregistry_in_bell_state()
        expected_state = np.array([[1/np.sqrt(2)],[0],[0],[1/np.sqrt(2)]])
        assert np.allclose(q.get_state(), expected_state, rtol=0, atol=tolerance), "Expected state is (2**-0.5)*(ket(00)+ket(11))"

    def test_apply_gate_sparse__should_work_for_simple_bell_state(self):
        q = self.__get_qregistry_in_bell_state()
        expected_state = np.array([[1/np.sqrt(2)],[0],[0],[1/np.sqrt(2)]])
        assert np.allclose(q.get_state(), expected_state, rtol=0, atol=tolerance), "Expected state is (2**-0.5)*(ket(00)+ket(11))"

    def test_value_prob__should_work_for_simple_bell_state(self):
        q = self.__get_qregistry_in_bell_state()
        expected_value_probs = [0.5,0,0,0.5]
        for value, _ in enumerate(expected_value_probs):
            assert np.allclose(q.value_prob(value), expected_value_probs[value], rtol=0, atol=tolerance), \
                (f"Expected probability of value {value} is not equal to the registry probability "
                 f"{q.value_prob(value)} != {expected_value_probs[value]}")

    def test_qbit_prob__should_work_for_simple_bell_state(self):
        q = self.__get_qregistry_in_bell_state()
        assert np.allclose(q.qbit_prob(0), 0.5, atol=tolerance)
        assert np.allclose(q.qbit_prob(1), 0.5, atol=tolerance)

    def test_qbit_prob__should_work_for_simple_state(self):
        q = qregistry.QRegistry(2)
        q.apply_gate(gates.x, 0)
        q.apply_gate(gates.h, 1)
        assert np.allclose(q.qbit_prob(0), 1, atol=tolerance)
        assert np.allclose(q.qbit_prob(1), 0.5, atol=tolerance)

    def test_collapse__when_no_probability_of_one_is_specified__should_work_for_simple_bell_state(self):
        q = self.__get_qregistry_in_bell_state()
        q.collapse(0,0)
        expected_state = np.array([[1],[0],[0],[0]])
        assert np.allclose(q.get_state(), expected_state, rtol=0, atol=tolerance), \
            "When the most significant qbit of Bell state (2**-0.5)*(ket(00)+ket(11)) is collapsed to 0, the other qbit must be 0 as well"

    def test_get_bloch_coords__should_work_for_initial_state(self):
        q = qregistry.QRegistry(1)
        expected_theta = 0
        expected_phi = 0
        assert np.allclose(q.get_bloch_coords()[0], expected_theta, atol=tolerance), "Theta should be 0 when qbit is 0"
        assert np.allclose(q.get_bloch_coords()[1], expected_phi, atol=tolerance), "Phi should be 0 when qbit is 0"

    def test_get_bloch_coords__should_work_after_applying_x_gate(self):
        q = qregistry.QRegistry(1)
        q.apply_gate(gates.x, 0)
        expected_theta = np.pi
        expected_phi = 0
        print(q.get_bloch_coords())
        assert np.allclose(q.get_bloch_coords()[0], expected_theta,
                           atol=tolerance), "Theta should be pi when qbit is 0"
        assert np.allclose(q.get_bloch_coords()[1], expected_phi,
                           atol=tolerance), "Phi should be 0 when qbit is 0"

    def test_get_bloch_coords__should_work_after_applying_x_and_h_gate(self):
        q = qregistry.QRegistry(1)
        q.apply_gate(gates.x, 0)
        q.apply_gate(gates.h, 0)
        expected_theta = np.pi/2
        expected_phi = np.pi
        assert np.allclose(q.get_bloch_coords()[0], expected_theta,
                           atol=tolerance), "Theta should be pi when qbit is 0"
        assert np.allclose(q.get_bloch_coords()[1], expected_phi,
                           atol=tolerance), "Phi should be 0 when qbit is 0"

    def test_measure__should_work_for_simple_bell_state(self):
        measurements = {0: 0, 1: 0}
        executions = 2000
        for _ in range(executions):
            q = self.__get_qregistry_in_bell_state()
            _, measured = q.measure(0)
            measurements[measured] = measurements.get(measured) + 1
        print(measurements[0]/executions)
        self.__assert_two_floats_are_close(
            50,
            measurements[0]/executions*100,
            message="About half of the values should have been 0",
            tol=5 #5%
        )

    def test_measure__should_work_for_simple_case(self):
        q = qregistry.QRegistry(2)
        q.apply_gate(gates.x, 0)
        q.apply_gate(gates.x, 1)
        assert q.measure(0)[1] == 1
        assert q.measure(1)[1] == 1

    def __assert_two_floats_are_close(self, expected, actual, message = None, tol = tolerance):
        if message is None:
            assert np.allclose(expected, actual, atol=tol)
        else:
            assert np.allclose(expected, actual, atol=tol), message + f". Expected value: {expected}, actual value: {actual}"

    def __get_qregistry_in_bell_state(self):
        q = qregistry.QRegistry(2)
        q.apply_gate_sparse(gates.h, 1)
        q.apply_gate_sparse(gates.cnot, 0)
        return q