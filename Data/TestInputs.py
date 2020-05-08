import numpy as np
from sklearn.preprocessing import MinMaxScaler

def normalize(x):
    return (x-np.mean(x, axis=0))/np.var(x, axis=0)

class XOR3D:
    def __init__(self):
        self.inputs = np.array([[0,0,0],[0,1,1],[1,0,1],[1,1,0]])
        self.targets = np.array([0, 1, 1, 0])

class XOR2D:
    def __init__(self):
        self.inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
        self.targets = np.array([0, 1, 1, 1])

class GaussSine:
    def __init__(self):
        scaler = MinMaxScaler()
        x = np.linspace(0,1,40).reshape(40, 1)
        t = np.sin(2*np.pi*x)+np.cos(4*np.pi*x) + np.random.randn(40,1)*0.2
        t_norm = normalize(t)
        scaler.fit(t)
        t_norm = scaler.transform(t)
        #x = x.T
        #t_norm = t_norm.T
        self.train = x[0::2,:]
        self.test = x[1::4,:]
        self.valid = x[3::4,:]
        self.trainTarget = t_norm[0::2,:]
        self.testTarget = t_norm[1::4,:]
        self.validTarget = t_norm[3::4,:]
