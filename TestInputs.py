import numpy as np

class XOR3D:
    def __init__(self):
        self.inputs = np.array([[0,0,0],[0,1,1],[1,0,1],[1,1,0]])
        self.targets = np.array([0, 1, 1, 0])

class XOR2D:
    def __init__(self):
        self.inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
        self.targets = np.array([0, 1, 1, 1])
