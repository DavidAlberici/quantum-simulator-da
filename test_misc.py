from unittest import TestCase

class TestMisc(TestCase):
    def test__parse_state_vector_positions_where_target_qbit_is_one(self):
        qbit = 2
        total_qbits = 3
        batch_size = int(2 ** (qbit + 1))
        half_batch_size = int(batch_size / 2)
        total_states = 2 ** total_qbits
        for batch_first_position in range(0, total_states, batch_size):
            first_value_one = batch_first_position + half_batch_size
            for j in range(first_value_one, first_value_one + half_batch_size, 1):
                print(j)

    def qbit_prob2(self, target):
        """
        Possible improvement to qbit_prob

        :param target:
        :return:
        """
        batch_size = int(2 ** (target + 1))
        half_batch_size = int(batch_size / 2)
        total_prob = 0.0
        for batch_first_position in range(0, self.state.size, batch_size):
            first_value_one = batch_first_position + half_batch_size
            for j in range(first_value_one, first_value_one + half_batch_size, 1):
                total_prob += self.value_prob(j)
        return total_prob