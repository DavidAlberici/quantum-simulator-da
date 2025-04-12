import gates
from qregistry import QRegistry

if __name__ == '__main__':
    print('PyCharm')

    num_qbits = 10
    # Este Qreg es el "nuevo", con scipy
    qreg = QRegistry(num_qbits)
    qreg.apply_gate(gates.x, num_qbits - 1)
    print(qreg.get_state()[1])
    print(qreg.measure(0)[1])
    # 0.2s para 14qbits

    # Este Qreg usa np.array, sin scipy
    # qreg = QRegistry3103(num_qbits)
    # qreg.apply_gate(gates.x, num_qbits - 1)
    # print(qreg.get_state()[1])
    # print(qreg.measure(0))
    # #7.8s para 14qbits

    ## Usar multiprocessing para intentar paralelizar!

