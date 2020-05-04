import numpy as np


def sigmoid(h, B):
    return 1/(1+np.exp(-B*h))

def sigmoid2(h):
    return ((np.exp(h) - np.exp(-h))/(np.exp(h)+np.exp(-h)))

def softmax(h):
    return np.exp(h)/(np.sum(np.exp(h)))

def linear(h):
    return h

class MultiLayerPerceptron:
    def __init__(self, inputs, targets, nHidden, beta=1, momentum=0.9, outtype="logistic"):
        self.inputs = inputs
        if len(np.shape(targets)) < 2:
            self.targets = targets.reshape((np.shape(targets)[0], 1))
        else:
            self.targets = targets

        self.nIn = np.shape(self.inputs)[1]
        self.nOut = np.shape(self.targets)[1]
        self.nData = np.shape(self.inputs)[0]
        self.nHidden = nHidden

        self.beta = beta
        self.momentum = momentum
        self.outtype = outtype

        self.weights1 = (np.random.rand(self.nIn+1, self.nHidden)-0.5)*2/np.sqrt(self.nIn)
        self.weights2 = (np.random.rand(self.nHidden+1, self.nOut)-0.5)*2/np.sqrt(self.nHidden)

        self.bias = -np.ones((self.nData, 1))
        self.hidden = None

    def train(self, eta, nIterations):
        inputs = np.concatenate((self.inputs, self.bias), axis=1)
        change = range(self.nData)

        updatew1 = np.zeros((np.shape(self.weights1)))
        updatew2 = np.zeros((np.shape(self.weights2)))

        for n in range(nIterations):
            outputs = self.forward(inputs, True)
            error = 0.5*np.sum((outputs-self.targets)**2)
            if (np.mod(n,100)==0):
                print(f"Iteration: {n}\tError: {error}")

            #output neuron types
            if self.outtype == "logistic":
                deltao = self.beta*(outputs-self.targets)*outputs*(1.0-outputs)
            else:
                raise ValueError("Invalid outtype")
            breakpoint()
            deltah = self.hidden*self.beta*(1.0-self.hidden)*(np.dot(deltao, np.transpose(self.weights2)))
            updatew1 = eta*(np.dot(np.transpose(inputs), deltah[:,:-1])) + self.momentum*updatew1
            updatew2 = eta*(np.dot(np.transpose(self.hidden), deltao)) + self.momentum*updatew2
            self.weights1 -= updatew1
            self.weights2 -= updatew2

    def forward(self, inputs, biased=False):
        if not biased:
            inputs = np.concatenate((inputs, -np.ones((np.shape(inputs)[0],1))), axis=1)
        self.hidden = np.dot(inputs, self.weights1)
        self.hidden = sigmoid(self.hidden, self.beta)
        self.hidden = np.concatenate((self.hidden, -np.ones((np.shape(inputs)[0],1))), axis=1)

        outputs = np.dot(self.hidden, self.weights2)

        if self.outtype == "logistic":
            return sigmoid(outputs, self.beta)

    def get_train_outputs(self):
        return self.forward(self.inputs)

    def conf_mat(self):
       """Confusion matrix"""

       # Add the inputs that match the bias node
       inputs = np.concatenate((self.inputs, self.bias),axis=1)
       outputs = self.forward(inputs, True)

       nclasses = np.shape(self.targets)[1]

       if nclasses==1:
           nclasses = 2
           outputs = np.where(outputs>0.5,1,0)
       else:
           # 1-of-N encoding
           outputs = np.argmax(outputs,1)
           targets = np.argmax(targets,1)

       cm = np.zeros((nclasses,nclasses))
       for i in range(nclasses):
           for j in range(nclasses):
               cm[i,j] = np.sum(np.where(outputs==i,1,0)*np.where(self.targets==j,1,0))

       print("Confusion matrix is:")
       print(cm)
       print(f"Percentage Correct: {np.trace(cm)/np.sum(cm)*100}")



class MultiLayerPerceptron2:
    def __init__(self, inputs, targets, nHidden=4, beta=1.0):
        # initialize inputs and targets
        try:
            inputs = np.array(inputs)
        except:
            raise ValueError("inputs arg must be a numpy array or numpy array-able")
        if len(np.shape(inputs)) == 1:
            self.inputs = inputs.reshape(np.shape(inputs)[0], 1)
        else:
            try:
                self.inputs = np.array([self.nparray(nested) for nested in inputs])
            except:
                raise ValueError("inputs arg must be a numpy array or numpy array-able")
        try:
            targets = np.array(targets)
        except:
            raise ValueError("targets arg must be a numpy array or numpy array-able")
        if len(np.shape(targets)) == 1:
            self.targets = targets.reshape(np.shape(targets)[0], 1)
        else:
            try:
                self.targets = np.array([self.nparray(nested) for nested in targets])
            except:
                raise ValueError("targets arg must be a numpy array or numpy array-able")

        nHidden = int(nHidden)
        if nHidden < 1:
            raise ValueError("nHidden must be an integer greater than 0")
        # initalize weights
        nIn = np.shape(self.inputs)[1]
        nOut = np.shape(self.targets)[1]
        self.weights_inputs = (np.random.rand(nIn, nHidden)-0.5)*2/np.sqrt(nIn)
        self.weights_hidden = (np.random.rand(nHidden, nOut)-0.5)*2/np.sqrt(nOut)

        # initialize bias
        self.bias_inputs = 0
        self.bias_hidden = 0

        if float(beta) <=0:
            raise ValueError("Beta must be greater than 0")
        self.beta = beta

    def train(self, eta, nIterations):
        inputs = self.inputs
        update_weights_inputs = np.zeros_like(self.weights_inputs)
        update_weights_hidden = np.zeros_like(self.weights_hidden)
        for i in range(nIterations):
            outputs = self.forward(inputs)
            error = np.sum((outputs-self.targets)**2)
            if (np.mod(n,100)==0):
                print(f"Iteration: {n}\tError: {error}")

            delta_output = 2*self.beta*(outputs-self.targets)*outputs*(1.0-outputs)
            delta_hidden =


    def forward(self, inputs):


        return outputs
