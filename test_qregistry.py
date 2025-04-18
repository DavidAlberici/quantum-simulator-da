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
        q = qregistry.QRegistry(2)
        q.apply_gate_sparse(gates.h, 1)
        q.apply_gate_sparse(gates.cnot, 0)
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
        seed = 1500
        rng = np.random.default_rng(seed)
        # Since we are using a seed, is not really necessary to do a "for" and test 2000 executions. But I initially
        # did it like that, and have not found the time to change it (and move this for to a "performance" test)
        for _ in range(executions):
            q = self.__get_qregistry_in_bell_state()
            _, measured = q.measure(0, rng)
            measurements[measured] = measurements.get(measured) + 1
        print(measurements[0]/executions)
        self.__assert_two_floats_are_close(
            50,
            measurements[0]/executions*100,
            message="About half of the values should have been 0",
            tol=5 #5%
        )

    def test_measure__should_work_for_eight_qbit_registry(self):
        # with this seed, .random() returns 0.6126 and 0.0158
        seed = 1001
        rng = np.random.default_rng(seed)
        # all qbits, in superposition, will have 50% chance of being 1. So with this seed the first measure must
        # be 0 (because random_num > prob_one [which is 50%=0.50]) and the second one 1 (random_num < prob_one)
        q = self.__get_qregistry_in_superposition(8)
        _, qbit0_measure = q.measure(0,rng)
        _, qbit1_measure = q.measure(1,rng)
        assert qbit0_measure == 0
        assert qbit1_measure == 1
        state = q.get_state()
        # Check the state is coherent. Since qbit 0 is 0, odd values are impossible. Also, since qbit 0 is 0, and
        # qbit 1 is 1, the only values possible are those for which: value mod 4 == 2. That means: [2, 6, 10, 14...]
        # The initial superposition had a 1/256 probability for every value. After collapsing two qbits, the
        # probability of remaining values is (1/64), which corresponds to an amplitude of (1/64)**0.5
        self.__assert_two_floats_are_close(0, state[0])
        self.__assert_two_floats_are_close(0, state[1])
        self.__assert_two_floats_are_close((1 / 64) ** 0.5, state[2])
        self.__assert_two_floats_are_close(0, state[3])
        self.__assert_two_floats_are_close((1 / 64) ** 0.5, state[6])
        self.__assert_two_floats_are_close((1 / 64) ** 0.5, state[82])

    def test_measure__should_work_for_simple_case(self):
        q = qregistry.QRegistry(2)
        q.apply_gate(gates.x, 0)
        q.apply_gate(gates.x, 1)
        assert q.measure(0)[1] == 1
        assert q.measure(1)[1] == 1

    def test_measure_parallel__should_work_for_simple_bell_state(self):
        measurements = {0: 0, 1: 0}
        executions = 2000
        seed = 1500
        rng = np.random.default_rng(seed)
        # Since we are using a seed, is not really necessary to do a "for" and test 2000 executions. But I initially
        # did it like that, and have not found the time to change it (and move this for to a "performance" test)
        for _ in range(executions):
            q = self.__get_qregistry_in_bell_state()
            _, measured = q.measure_parallel(0, rng)
            measurements[measured] = measurements.get(measured) + 1
        print(measurements[0]/executions)
        self.__assert_two_floats_are_close(
            50,
            measurements[0]/executions*100,
            message="About half of the values should have been 0",
            tol=5 #5%
        )

    def test_measure_parallel__should_work_for_eight_qbit_registry(self):
        # with this seed, .random() returns 0.6126 and 0.0158
        seed = 1001
        rng = np.random.default_rng(seed)
        # all qbits, in superposition, will have 50% chance of being 1. So with this seed the first measure must
        # be 0 (because random_num > prob_one [which is 50%=0.50]) and the second one 1 (random_num < prob_one)
        q = self.__get_qregistry_in_superposition(8)
        _, qbit0_measure = q.measure_parallel(0,rng, 3)
        _, qbit1_measure = q.measure_parallel(1,rng, 3)
        assert qbit0_measure == 0
        assert qbit1_measure == 1
        state = q.get_state()
        # Check the state is coherent. Since qbit 0 is 0, odd values are impossible. Also, since qbit 0 is 0, and
        # qbit 1 is 1, the only values possible are those for which: value mod 4 == 2. That means: [2, 6, 10, 14...]
        # The initial superposition had a 1/256 probability for every value. After collapsing two qbits, the
        # probability of remaining values is (1/64), which corresponds to an amplitude of (1/64)**0.5
        self.__assert_two_floats_are_close(0, state[0])
        self.__assert_two_floats_are_close(0, state[1])
        self.__assert_two_floats_are_close((1 / 64) ** 0.5, state[2])
        self.__assert_two_floats_are_close(0, state[3])
        self.__assert_two_floats_are_close((1 / 64) ** 0.5, state[6])
        self.__assert_two_floats_are_close((1 / 64) ** 0.5, state[82])

    def test_measure_parallel__should_work_for_simple_case(self):
        q = qregistry.QRegistry(2)
        q.apply_gate(gates.x, 0)
        q.apply_gate(gates.x, 1)
        assert q.measure_parallel(0)[1] == 1
        assert q.measure_parallel(1)[1] == 1

    def __assert_two_floats_are_close(self, expected, actual, message = None, tol = tolerance):
        if message is None:
            assert np.allclose(expected, actual, atol=tol)
        else:
            assert np.allclose(expected, actual, atol=tol), message + f". Expected value: {expected}, actual value: {actual}"

    def __get_qregistry_in_bell_state(self):
        q = qregistry.QRegistry(2)
        q.apply_gate(gates.h, 1)
        q.apply_gate(gates.cnot, 0)
        return q

    def __get_qregistry_in_superposition(self, num_qbits):
        q = qregistry.QRegistry(num_qbits)
        for i in range(num_qbits):
            q.apply_gate(gates.h,i)
        return q