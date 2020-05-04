import numpy as np

class LinearRegression:
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.bias = -np.ones((np.shape(inputs)[0], 1))
        self.biased_inputs = np.concatenate((self.inputs, self.bias), axis=1)
        self.targets = targets
        self.beta = None

    def train(self):
        self.beta = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(self.biased_inputs),self.biased_inputs)),np.transpose(self.biased_inputs)),self.targets)

    def get_train_output(self):
        return np.dot(self.biased_inputs, self.beta)
