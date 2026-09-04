from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
def grover_search_oracle_00() ->QuantumCircuit:
    
    qr = QuantumRegister(2, 'q')
    cr = ClassicalRegister(2, 'c')
    qc = QuantumCircuit(qr, cr)

    qc.h([0, 1])
    qc.x([0, 1])
    qc.cz(0, 1)
    qc.x([0, 1])
    qc.h([0, 1])
    qc.z([0, 1])
    qc.cz(0, 1)
    qc.x([0, 1])
    qc.h([0, 1])
        
    qc.measure([0, 1], [0, 1])
   
    return qc
