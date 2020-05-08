import numpy as np

def slp(inputs, targets, **kwargs):

    def train():
        for n in range(iterations):
            backward(n)

    def backward(n):
        nonlocal up_wts, wts, bias
        outputs = forward(inputs)

        if not n%100:
            print(f"Iteration: {n}\tError: {np.sum((outputs-targets)**2)}")

        delta_out = d_activation_functions[outtype](outputs)

        up_wts = eta*np.dot(inputs.T, delta_out)+momentum*up_wts

        wts -=up_wts

        bias -= eta*(np.sum(delta_out, axis=0).reshape(nOut, 1))
        return

    def forward(inputs):
        outputs = activation_functions[outtype](np.dot(inputs, wts).T + bias)
        return outputs.T

    def numpy_arrayify(x):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
            x = np.array([np.array(nest) for nest in x])
        if isinstance(x, np.ndarray) and len(np.shape(x)) == 1:
            x = x.reshape(len(x), 1)
        return x

    def raise_value_error(msg):
        raise ValueError(msg)

    beta = kwargs.get("beta", 1.0)
    momentum = kwargs.get("momentum", 0.9)
    seed = kwargs.get("seed", 42)
    eta = kwargs.get("eta", 0.25)
    iterations = int(kwargs.get("iterations", 1000))
    outtype = kwargs.get("outtype", "logistic")

    inputs = numpy_arrayify(inputs)
    targets = numpy_arrayify(targets)

    nData = np.shape(inputs)[0]
    nIn = np.shape(inputs)[1]
    nOut = np.shape(targets)[1]

    np.random.seed(seed)
    wts = (np.random.rand(nIn, nOut))*0.1-0.05

    up_wts = np.zeros_like(wts)

    bias = np.zeros((nOut, 1))

    def soft_max(x):
        normalisers = np.sum(np.exp(x).T, axis=1).reshape(1, np.shape(x)[1])
        return np.exp(x)/normalisers

    activation_functions = {"linear": lambda x : x,
                            "logistic": lambda x : 1.0/(1.0+np.exp(-beta*x)),
                            "softmax": soft_max
                            }

    d_activation_functions = {"linear": lambda x : (x-targets)/nData,
                              "logistic": lambda x : beta*(x-targets)*x*(1.0-x),
                              "softmax": lambda x : (x-targets)*(x*(-x)+x)/nData
                              }

    train()

    return lambda x: forward(x)
