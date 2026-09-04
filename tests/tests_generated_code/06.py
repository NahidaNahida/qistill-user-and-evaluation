from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
def swaptest_zaxis(unknown_state: QuantumCircuit) ->QuantumCircuit: 

    qr = QuantumRegister(3, 'q')  
    cr = ClassicalRegister(1, 'c') 
    qc = QuantumCircuit(qr, cr)
    
    qc = qc.compose(unknown_state, [qr[1]])  
    qc.h(qr[0])
    qc.cswap(qr[0], qr[1], qr[2]) 
    qc.h(qr[0])
    qc.measure(qr[0], cr[0])
    return qc
