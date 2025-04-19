from unittest import TestCase
import numpy as np
import scipy as sp
import gates

class TestMisc(TestCase):
    """
    This class can be ignored, is just like a "notebook" to make drafts/sketches
    """
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

    @staticmethod
    def basic_sparse_array_methods():
        arr = sp.sparse.csr_array((2, 2))  # create 2x2 matrix
        arr[(1, 0)] = 1  # assign
        print(arr)
        print("*******")
        print(arr.toarray())  # convert to normal array
        print("*******")
        nparr = np.eye(8)
        scarr = sp.sparse.csr_array(nparr)  # support conversion from np to sp.csr
        print(scarr)

    @staticmethod
    def kron_in_sparse_array():
        h_sparse = sp.sparse.csr_array(gates.h)
        i_sparse = sp.sparse.csr_array(gates.I(1))
        print(sp.sparse.kron(h_sparse, i_sparse))
        # print(np.kron(h_sparse, i_sparse))