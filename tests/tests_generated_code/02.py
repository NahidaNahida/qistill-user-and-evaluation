from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit

def generate_quantum_state_qubit3() -> QuantumCircuit:
    qc = QuantumCircuit(3,3)   
    qc.h(2)     
    qc.x(1)     
    qc.cx(2,1)  
    qc.cx(1,0)
    qc.z(2) 
    qc.measure([0,1,2], [0,1,2])
    
    return qc
