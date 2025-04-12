import numpy as np

class Gates:
    x = np.array([[0,1],[1,0]], dtype=np.complex64)
    y = np.array([[0,-0j],[0j,0]], dtype=np.complex64)
    z = np.array([[1,0],[0,-1]], dtype=np.complex64)
    h = np.array([[(2**-0.5),(2**-0.5)],[(2**-0.5),-(2**-0.5)]], dtype=np.complex64)
    swap = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=np.complex64)
    I = lambda nq : np.eye(2**nq, dtype=np.complex64)
    rx = lambda phi : np.array([[np.cos(phi/2),-np.sin(phi/2)*1j],[-np.sin(phi/2)*1j, np.cos(phi/2)]], dtype=np.complex64)
    ry = lambda phi : np.array([[np.cos(phi/2),-np.sin(phi/2)],[np.sin(phi/2), np.cos(phi/2)]], dtype=np.complex64)
    rz = lambda phi : np.array([[np.e**(-phi/2*1j),0],[0,np.e**(phi/2*1j)]], dtype=np.complex64)
    cnot = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=np.complex64)

x = np.array([[0,1],[1,0]], dtype=np.complex64)
y = np.array([[0,-0j],[0j,0]], dtype=np.complex64)
z = np.array([[1,0],[0,-1]], dtype=np.complex64)
h = np.array([[(2**-0.5),(2**-0.5)],[(2**-0.5),-(2**-0.5)]], dtype=np.complex64)
swap = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=np.complex64)
I = lambda nq : np.eye(2**nq, dtype=np.complex64)
rx = lambda phi : np.array([[np.cos(phi/2),-np.sin(phi/2)*1j],[-np.sin(phi/2)*1j, np.cos(phi/2)]], dtype=np.complex64)
ry = lambda phi : np.array([[np.cos(phi/2),-np.sin(phi/2)],[np.sin(phi/2), np.cos(phi/2)]], dtype=np.complex64)
rz = lambda phi : np.array([[np.e**(-phi/2*1j),0],[0,np.e**(phi/2*1j)]], dtype=np.complex64)
cnot = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=np.complex64)