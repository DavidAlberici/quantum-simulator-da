import numpy as np
import scipy as sp
import gates

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


def kron_in_sparse_array():
    h_sparse = sp.sparse.csr_array(gates.h)
    i_sparse = sp.sparse.csr_array(gates.I(1))
    print(sp.sparse.kron(h_sparse, i_sparse))
    # print(np.kron(h_sparse, i_sparse))


if __name__ == '__main__':
    print('PyCharm')