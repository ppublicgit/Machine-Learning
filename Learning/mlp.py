import numpy as np


def mlp(inputs, targets, **kwargs):

    def forward(inputs):
        hidden = activation_functions["logistic"](np.dot(inputs, wt_in).T+b_in)
        outputs = activation_functions[outtype](np.dot(hidden.T, wt_hn).T+b_hn)
        return outputs.T, hidden.T


    def backward(n):
        nonlocal wt_in, wt_hn, up_wt_in, up_wt_hn, b_in, b_hn
        outputs, hidden = forward(inputs)
        if not n%100:
            print(f"Iteration: {n}\tError: {np.sum((outputs-targets)**2)}")

        delta_out = d_activation_functions[outtype](outputs)
        delta_hdn = hidden*beta*(1-hidden)*np.dot(delta_out, wt_hn.T)

        up_wt_in = eta*np.dot(inputs.T, delta_hdn)+momentum*up_wt_in
        up_wt_hn = eta*np.dot(hidden.T, delta_out)+momentum*up_wt_hn

        wt_in -= up_wt_in
        wt_hn -= up_wt_hn

        b_in -= eta*(np.sum(delta_hdn, axis=0).reshape(nHidden, 1))
        b_hn -= eta*(np.sum(delta_out, axis=0).reshape(nOut, 1))

    def train():
        if validate:
            count=0
            new_error, prev_error1, prev_error2 = 999, np.inf, 1000
            while ((prev_error1 - new_error)>0.001 or (prev_error2 - prev_error1)>0.001):
                count += 1
                print(count)
                for i in range(valid_iterations):
                    backward(i)
                prev_error2 = prev_error1
                prev_error1 = new_error
                valid_out, _ = forward(valid_inputs)
                new_error = 0.5*np.sum((valid_targets-valid_out)**2)
            print(f"Stoppped... Total Iterations: {count*valid_iterations}")
            print(f"Final three errors: {new_error}, {prev_error1}, {prev_error2}")

        else:
            for i in range(iterations):
                backward(i)

    def numpy_arrayify(x):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
            x = np.array([np.array(nest) for nest in x])
        if isinstance(x, np.ndarray) and len(np.shape(x)) == 1:
            x = x.reshape(len(x), 1)
        return x

    def raise_value_error(msg):
        raise ValueError(msg)

    nHidden = int(kwargs.get("nHidden", 4))
    beta = kwargs.get("beta", 1.0)
    momentum = kwargs.get("momentum", 0.9)
    outtype = kwargs.get("outtype", "logistic").lower()
    seed = kwargs.get("seed", 42)
    eta = kwargs.get("eta", 0.25)
    iterations = int(kwargs.get("iterations", 1000))
    validate = bool(kwargs.get("validate", False))
    if validate:
        valid_inputs = numpy_arrayify(kwargs["valid_inputs"]) if "valid_inputs" in kwargs else \
            raise_value_error("valid inputs not set")
        valid_targets = numpy_arrayify(kwargs["valid_targets"]) if "valid_targets" in kwargs else \
                                                  raise_value_error("value targets not set")
        valid_iterations = int(kwargs.get("valid_iterations", 100))

    inputs = numpy_arrayify(inputs)
    targets = numpy_arrayify(targets)

    nData = np.shape(inputs)[0]
    nIn = np.shape(inputs)[1]
    nOut = np.shape(targets)[1]

    np.random.seed(seed)
    wt_in = (np.random.rand(nIn, nHidden)-0.5)*2/np.sqrt(nIn)
    wt_hn = (np.random.rand(nHidden, nOut)-0.5)*2/np.sqrt(nHidden)

    #wt_in = (np.random.rand(nIn+1, nHidden)-0.5)*2/np.sqrt(nIn)
    #wt_hn = (np.random.rand(nHidden+1, nOut)-0.5)*2/np.sqrt(nHidden)


    up_wt_in = np.zeros_like(wt_in)
    up_wt_hn = np.zeros_like(wt_hn)

    b_in = np.zeros((nHidden, 1))
    b_hn = np.zeros((nOut, 1))

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

    return lambda x : forward(x)[0]
