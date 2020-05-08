import numpy as np

from .slp import slp


def rbf(inputs, targets, **kwargs):

    def train():
        nonlocal wts, hidden, slp_fwd, wts_fwd
        if use_kmeans:
            raise Exception("kmeans not supported yet")
            #wts = np.transpose(kmeansnet.kmeanstrain(inputs))
        else:
            indices = [*range(nData)]
            np.random.shuffle(indices)
            for i in range(nRbf):
                wts[:, i] = inputs[indices[i], :]

        for i in range(nRbf):
            hidden[:, i] = np.exp(-np.sum((inputs - np.ones((1, nIn))*wts[:, i])**2, axis=1) \
                                  / (2*sigma**2))

        if normalize:
            hidden /= (np.ones((1, nData)) * hidden.sum(axis=1)).T

        if outtype == "perceptron":
            slp_fwd = slp(hidden, targets, eta=eta, iterations=iterations)
        elif outtype == "weight_matrix":
            hidden = np.concatenate((hidden, -np.ones((nData, 1))), axis=1)
            wts_fwd = np.dot(np.linalg.pinv(hidden), targets)
        else:
            raise ValueError(f"Invalid outtype: {outtype}")

    def forward(inputs):
        hidden = np.zeros((inputs.shape[0], nRbf))
        for i in range(nRbf):
            hidden[:, i] = np.exp(-np.sum((inputs - np.ones((1, nIn))*wts[:, i])**2, axis=1) \
                                  / (2*sigma**2))

        if normalize:
            hidden /= (np.ones((1, inputs.shape[0])) * hidden.sum(axis=1)).T

        if outtype == "perceptron":
            outputs = slp_fwd(hidden)
        elif outtype == "weight_matrix":
            hidden = np.concatenate((hidden, -np.ones((inputs.shape[0], 1))), axis=1)
            outputs = np.dot(hidden, wts_fwd)
        else:
            raise ValueError(f"Invalid outtype: {outtype}")
        return outputs


    def numpy_arrayify(x):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
            x = np.array([np.array(nest) for nest in x])
        if isinstance(x, np.ndarray) and len(np.shape(x)) == 1:
            x = x.reshape(len(x), 1)
        return x

    eta = kwargs.get("eta", 0.25)
    iterations = int(kwargs.get("iterations", 1000))
    use_kmeans = bool(kwargs.get("use_kmeans", False))
    nRbf = int(kwargs.get("nRbf", 4))
    sigma = float(kwargs.get("sigma", 0))
    normalize = bool(kwargs.get("normalize", True))
    outtype = kwargs.get("outtype", "perceptron")

    inputs = numpy_arrayify(inputs)
    targets = numpy_arrayify(targets)

    nData = np.shape(inputs)[0]
    nIn = np.shape(inputs)[1]
    nOut = np.shape(targets)[1]

    hidden = np.zeros((nData, nRbf))

    if use_kmeans:
        raise Exception("kmeans not supported yet")
        #kmeansnet = kmeans.kmeans(nRbf, inputs)

    if sigma == float(0):
        d = (inputs.max(axis=0) - inputs.min(axis=0)).max()
        sigma = d / np.sqrt(2 * nRbf)

    wts = np.zeros((nIn, nRbf))
    wts_fwd = np.zeros((nRbf+1, nOut))
    slp_fwd = None

    train()

    return (lambda x: forward(x))
