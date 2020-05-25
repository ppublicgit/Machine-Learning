import os
import numpy as np
import matplotlib.pyplot as plt
from Learning.unsupervised import kmeans, kmeansnet

def read_iris(filepath):
    iris = np.loadtxt(filepath, delimiter=',')
    iris[:,:4] = iris[:,:4]-iris[:,:4].mean(axis=0)
    imax = np.concatenate((iris.max(axis=0)*np.ones((1,5)),iris.min(axis=0)*np.ones((1,5))),axis=0).max(axis=0)
    iris[:,:4] = iris[:,:4]/imax[:4]

    target = iris[:,4]

    order = [*range(np.shape(iris)[0])]
    np.random.shuffle(order)
    iris = iris[order,:]
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

    cluster = kmeans(x_in, 3, average="mean")
    predict = cluster(y_in)

    cluster_net = kmeansnet(x_in, 3)
    predict_net = cluster_net(y_in)

    print(f"KMEANS    : {predict[:, 0]}")
    print(f"KMEANS NET: {predict_net[:, 0]}")
    print(f"ACTUAL    : {y_out}")

    print(f"KMEANS DIFFS COUNTS     : \n{create_mapping(predict[:, 0], y_out)}")
    print(f"KMEANS NET DIFFS COUNTS : \n{create_mapping(predict_net[:, 0], y_out)}")
