import os
import numpy as np
import matplotlib.pyplot as plt
from Learning.unsupervised import kmeans, kmeansnet, som

def read_iris(filepath):
    iris = np.loadtxt(filepath, delimiter=',')
    iris[:, :4] = iris[:, :4]-iris[:, :4].mean(axis=0)
    imax = np.concatenate((iris.max(axis=0)*np.ones((1, 5)),
                           iris.min(axis=0)*np.ones((1, 5))), axis=0).max(axis=0)
    iris[:, :4] = iris[:, :4]/imax[:4]

    target = iris[:, 4]

    order = [*range(np.shape(iris)[0])]
    np.random.shuffle(order)
    iris = iris[order, :]
    target = target[order]

    return iris, target


def create_mapping(predictions, targets):
    mapping = np.zeros((3, 3))
    for i in range(len(predictions)):
        mapping[predictions[i], targets[i]] += 1
    return mapping


if __name__ == "__main__":
    data, targets = read_iris(os.path.join(os.getcwd(), "Data/iris_proc.data"))

    x_in = data[::2, 0:4]
    x_out = targets[::2]
    y_in = data[1::2, 0:4]
    y_out = targets[1::2].astype(int)

    train = x_in
    traint = x_out
    test = y_in
    testt = y_out

    cluster = kmeans(x_in, 3, average="mean")
    predict = cluster(y_in)

    cluster_net = kmeansnet(x_in, 3)
    predict_net = cluster_net(y_in)

    print(f"KMEANS    : {predict[:, 0]}")
    print(f"KMEANS NET: {predict_net[:, 0]}")
    print(f"ACTUAL    : {y_out}")

    print(
        f"KMEANS DIFFS COUNTS     : \n{create_mapping(predict[:, 0], y_out)}")
    print(
        f"KMEANS NET DIFFS COUNTS : \n{create_mapping(predict_net[:, 0], y_out)}")

    forward = som(x_in, (6, 6), nIterations=400)
    som_map = forward("getMap")
    mapped = np.zeros(y_in.shape[0], dtype=int)

    for i in range(len(y_in)):
        mapped[i] = forward(x_in[i, :])
    plt.figure()
    plt.title("Trained SOM Map")
    plt.plot(som_map[0, :], som_map[1, :], 'k.', ms=15)
    where = np.argwhere(x_out == 0).flatten()
    plt.plot(som_map[0, mapped[where]], som_map[1, mapped[where]], 'rs', ms=30, label="0")
    where = np.argwhere(x_out == 1).flatten()
    plt.plot(som_map[0, mapped[where]], som_map[1, mapped[where]], 'gv', ms=30, label="1")
    where = np.argwhere(x_out == 2).flatten()
    plt.plot(som_map[0, mapped[where]], som_map[1, mapped[where]], 'b^', ms=30, label="2")
    plt.axis([-0.1, 1.1, -0.1, 1.1])
    plt.axis('off')
    plt.show(block=False)

    mapped = np.zeros(y_in.shape[0], dtype=int)
    for i in range(len(y_in)):
        mapped[i] = forward(y_in[i, :])

    plt.figure()
    plt.title("Tested SOM Map")
    plt.plot(som_map[0, :], som_map[1, :], 'k.', ms=15)
    where = np.argwhere(y_out == 0).flatten()
    plt.plot(som_map[0, mapped[where]], som_map[1, mapped[where]], 'rs', ms=30, label="0")
    where = np.argwhere(y_out == 1).flatten()
    plt.plot(som_map[0, mapped[where]], som_map[1, mapped[where]], 'gv', ms=30, label="1")
    where = np.argwhere(y_out == 2).flatten()
    plt.plot(som_map[0, mapped[where]], som_map[1, mapped[where]], 'b^', ms=30, label="2")
    plt.axis([-0.1, 1.1, -0.1, 1.1])
    plt.axis('off')
    plt.show(block=True)
