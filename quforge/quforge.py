import torch
import quforge.aux as aux
import quforge.gates as gates
import quforge.statevector as sv
import quforge.circuit as circuit
import quforge.optimizer as optimizer

def State(*args, **kwargs):
    return sv.State(*args, **kwargs)

def I(*args, **kwargs):
    return gates.I(*args, **kwargs)

def H(*args, **kwargs):
    return gates.H(*args, **kwargs)

def X(*args, **kwargs):
    return gates.X(*args, **kwargs)

def Y(*args, **kwargs):
    return gates.Y(*args, **kwargs)

def Z(*args, **kwargs):
    return gates.Z(*args, **kwargs)

def RX(*args, **kwargs):
    return gates.RX(*args, **kwargs)

def RY(*args, **kwargs):
    return gates.RY(*args, **kwargs)

def RZ(*args, **kwargs):
    return gates.RZ(*args, **kwargs)

def CNOT(*args, **kwargs):
    return gates.CNOT(*args, **kwargs)

def SWAP(*args, **kwargs):
    return gates.SWAP(*args, **kwargs)

def CZ(*args, **kwargs):
    return gates.CZ(*args, **kwargs)

def CRX(*args, **kwargs):
    return gates.CRX(*args, **kwargs)

def CRY(*args, **kwargs):
    return gates.CRX(*args, **kwargs)

def CRZ(*args, **kwargs):
    return gates.CRZ(*args, **kwargs)

def CU(*args, **kwargs):
    return gates.CU(*args, **kwargs)

def CCNOT(*args, **kwargs):
    return gates.CCNOT(*args, **kwargs)

def MCX(*args, **kwargs):
    return gates.MCX(*args, **kwargs)

def U(*args, **kwargs):
    return gates.U(*args, **kwargs)

def Circuit(*args, **kwargs):
    return circuit.Circuit(*args, **kwargs)

def measure(*args, **kwargs):
    return sv.measure(*args, **kwargs)

def optim(*args, **kwargs):
    return optimizer.optim(*args, **kwargs)
optim.Adam = optimizer.optim.Adam
optim.SGD  = optimizer.optim.SGD

def sum(*args, **kwargs):
    return torch.sum(*args, **kwargs)

def kron(*args, **kwargs):
    return aux.kron(*args, **kwargs)