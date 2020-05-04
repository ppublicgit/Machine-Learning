import numpy as np

class Perceptron:
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.nIn, self.nData, self.bias = self._init_input(inputs)
        if len(np.shape(targets)) == 1:
            self.targets = targets.reshape((np.shape(targets)[0], 1))
        else:
            self.targets = targets
        self.nOut = np.shape(targets)[1] if np.ndim(targets)>1 else 1
        self.weights = np.random.rand(self.nIn+1, self.nOut)*0.1 -0.5

    def _init_input(self, inputs):
        nIn = np.shape(inputs)[1] if np.ndim(inputs)>1 else 1
        nData = np.shape(inputs)[0]
        bias = -np.ones((nData,1))
        return nIn, nData, bias

    def train(self, eta, nIterations):
        if self.inputs is None:
            raise ValueError("Inputs are not set. set inputs on initialization or with set_test_data")
        inputs = np.concatenate((self.inputs, self.bias), axis=1)
        change = range(self.nData)
        for n in range(nIterations):
            self.activations = self.forward(inputs, biased=True)
            self.weights -= eta*np.dot(np.transpose(inputs), self.activations - self.targets)

    def forward(self, inputs, biased=False):
        if biased == False:
            _, _, bias = self._init_input(inputs)
            inputs = np.concatenate((inputs, bias), axis=1)
        activations = np.dot(inputs, self.weights)
        return np.where(activations>0,1,0)

    def print_outputs(self):
        inputs = np.concatenate((self.inputs, self.bias), axis=1)
        outputs = np.dot(inputs, self.weights)
        print(outputs)

    def conf_mat(self):
        inputs = np.concatenate((self.inputs, self.bias), axis=1)
        outputs = np.dot(inputs, self.weights)
        targets = self.targets
        nClasses = np.shape(targets)[1]
        if nClasses == 1:
            nClasses = 2
            outputs = np.where(outputs>0,1,0)
        else:
            outputs = np.argmax(outputs, 1)
            targets = np.argmax(targets, 1)
        cm = np.zeros((nClasses, nClasses))
        for i in range(nClasses):
            for j in range(nClasses):
                cm[i,j] = np.sum(np.where(outputs==i,1,0)*np.where(targets==j,1,0))
        print(cm)
        print(np.trace(cm)/np.sum(cm))
        return cm
