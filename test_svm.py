from Learning.svm import svm
import numpy as np
import matplotlib.pyplot as plt


def modified_XOR(kernel, degree, C, sdev):
    m = 100
    X = sdev*np.random.randn(m, 2)
    X[m//2:, 0] += 1.
    X[m//4:m//2, 1] += 1.
    X[3*m//4:, 1] += 1.
    targets = -np.ones((m, 1))
    targets[:m//4, 0] = 1.
    targets[3*m//4:, 0] = 1.

    classifier = svm(X, targets, kernel=kernel, degree=degree, C=C)

    Y = sdev*np.random.randn(m, 2)
    Y[m//2:, 0] += 1.
    Y[m//4:m//2, 1] += 1.
    Y[3*m//4:m, 1] += 1.
    test = -np.ones((m, 1))
    test[:m//4, 0] = 1.
    test[3*m//4:, 0] = 1.

    output, sv = classifier(Y, soft=False, get_sv=True)

    err1 = np.where((output == 1.) & (test == -1.))[0]
    err2 = np.where((output == -1.) & (test == 1.))[0]
    print(kernel, C)
    print(f"Class 1 errors {len(err1)} from {len(test[test == 1])}")
    print(f"Class 2 errors {len(err2)} from {len(test[test == -1])}")
    print(f"Test accuracy {1. - (float(len(err1)+len(err2))) / (len(test[test == 1]) + len(test[test == -1]))}")

    plt.figure()
    l1 = np.where(targets == 1)[0]
    l2 = np.where(targets == -1)[0]
    plt.plot(X[sv, 0], X[sv, 1], 'o', markeredgewidth=5)
    plt.plot(X[l1, 0], X[l1, 1], 'ro')
    plt.plot(X[l2, 0], X[l2, 1], 'yo')
    l1 = np.where(test == 1)[0]
    l2 = np.where(test == -1)[0]
    plt.plot(Y[l1, 0], Y[l1, 1], 'rs')
    plt.plot(Y[l2, 0], Y[l2, 1], 'ys')
    step = 0.1
    f0, f1 = np.meshgrid(np.arange(np.min(X[:, 0])-0.5, np.max(
        X[:, 0])+0.5, step), np.arange(np.min(X[:, 1])-0.5, np.max(X[:, 1])+0.5, step))

    out = classifier(np.c_[np.ravel(f0), np.ravel(f1)], soft=True).T
    out = out.reshape(f0.shape)
    plt.contour(f0, f1, out, 2)

    plt.axis('off')
    plt.title(f"{kernel}\n{degree}\n{C}\n{sdev}")
    plt.show(block=False)


if __name__ == "__main__":
    np.random.seed(0)
    for sdev in [0.1, 0.3, 0.4]:
        modified_XOR('linear', 1, None, sdev)
        modified_XOR('linear', 1, 0.1, sdev)
        modified_XOR('poly', 3, None, sdev)
        modified_XOR('poly', 3, 0.1, sdev)
        modified_XOR('rbf', 0, None, sdev)
        modified_XOR('rbf', 0, 0.1, sdev)
    input("Press enter to exit...")
