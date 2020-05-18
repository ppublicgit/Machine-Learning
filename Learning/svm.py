import numpy as np
import cvxopt

def svm(inputs, targets, **kwargs):

    def build_kernel(inputs):
        nonlocal K, inputs_squared
        K = np.dot(inputs, inputs.T).reshape(nData, nData)
        if kernel == "poly":
            K = (1.0 + 1.0/sigma * K)**degree
        elif kernel == "rbf":
            inputs_squared = (np.diag(K)*np.ones((1, nData))).T
            b = np.ones((nData, 1))
            K -= 0.5*(np.dot(inputs_squared, b.T) + (np.dot(b, inputs_squared.T)))
            K = np.exp(K/(sigma**2)) # 1/2 or not ?

    def train():
        nonlocal inputs, targets, b, lambdas, n_support, support_vectors
        build_kernel(inputs)
        P = targets * targets.T * K
        q = -np.ones((nData, 1))
        if C is None:
            G = -np.eye(nData)
            h = np.zeros((nData, 1))
        else:
            G = np.concatenate((np.eye(nData), -np.eye(nData)))
            h = np.concatenate((C * np.ones((nData, 1)), np.zeros((nData, 1))))
        A = targets.reshape(1, nData)
        if isinstance(A[0][0], np.integer):
            print(("Warning, type of targets for svm must be float in order for "
                  "quadratic solver cvxopt to work. Converting type of targets from"
                   " int to float."))
            A = A.astype(float)
        b = 0.0

        solved = cvxopt.solvers.qp(cvxopt.matrix(P), cvxopt.matrix(q), cvxopt.matrix(G),
                                    cvxopt.matrix(h), cvxopt.matrix(A), cvxopt.matrix(b))

        lambdas = np.array(solved["x"])

        support_vectors = np.where(lambdas > threshold)[0]
        n_support = len(support_vectors)
        print(f"{n_support} support vectors found")

        inputs = inputs[support_vectors, :]
        lambdas = lambdas[support_vectors]
        targets = targets[support_vectors]

        b = np.sum(targets)
        for n in range(n_support):
            b -= np.sum(lambdas * targets *
                        np.reshape(K[support_vectors[n], support_vectors], (n_support, 1)))
        b /= float(n_support)

    def classifier(Y, soft=False, get_sv=False):
        if kernel == "poly":
            K = (1.0 + 1.0/sigma * np.dot(Y, inputs.T))**degree
            y = np.zeros((Y.shape[0], 1))
            for j in range(Y.shape[0]):
                for i in range(n_support):
                    y[j] += lambdas[i] * targets[i] * K[j, i]
                y[j] += b

        elif kernel == "rbf":
            K = np.dot(Y, inputs.T)
            c = (1.0/sigma * np.sum(Y**2, axis=1)).reshape(Y.shape[0], 1)
            c = np.dot(c, np.ones((1, K.shape[1])))
            aa = np.dot(inputs_squared[support_vectors], np.ones((1, K.shape[0]))).T
            K = K - 0.5 * c -0.5 * aa
            K = np.exp(K/(sigma**2)) # 1/2 or not?

            y = np.zeros((Y.shape[0], 1))
            for j in range(Y.shape[0]):
                for i in range(n_support):
                    y[j] += lambdas[i] * targets[i] * K[j, i]
                y[j] += b

        if soft:
            ret = y
        else:
            ret = np.sign(y)

        if get_sv:
            return ret, support_vectors
        else:
            return ret

    def numpy_arrayify(x):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
            x = np.array([np.array(nest) for nest in x])
        if isinstance(x, np.ndarray) and len(np.shape(x)) == 1:
            x = x.reshape(len(x), 1)
        return x

    kernel = kwargs.get("kernel", "linear").lower()
    C = kwargs.get("C", None)
    sigma = kwargs.get("sigma", 1.0)
    degree = kwargs.get("degree", None)
    threshold = kwargs.get("threshold", 1e-5)

    if kernel not in ["poly", "rbf", "linear"]:
        raise ValueError((f"Invalid kernel specified {kernel}. "
                          "Must be either linear, poly or rbf"))

    inputs = numpy_arrayify(inputs)
    targets = numpy_arrayify(targets)

    nData = inputs.shape[0]

    if kernel == "linear":
        degree = 1
        kernel = "poly"
    if kernel == "poly" and degree is None:
        raise ValueError("degree must be set if using poly kernel")

    K = None
    b = None
    n_support = None
    lambdas = None
    inputs_squared = None
    support_vectors = None

    train()

    return classifier
