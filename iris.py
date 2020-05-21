import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy
from time import time

from Learning.scoring import conf_mat
from Learning.mlp import mlp
from Learning.rbf import rbf
from Learning.svm import svm
from Learning.probabilistic_learning import knn
from Learning.decision_trees import dtree

def read_data():
    df = pd.read_csv(os.path.join(os.getcwd(), "Data/iris.data"), header=None)
    replace = [("Iris-setosa", 0), ("Iris-versicolor", 1),
               ("Iris-virginica", 2)]
    [df.replace(to_replace=replace[i][0], value=replace[i][1], inplace=True)
     for i in range(len(replace))]
    df = df.astype({4: int})
    #df_norm = (df.iloc[:, 0:-1]-df.iloc[:, 0:-1].mean())/df.iloc[:, 0:-1].max()
    #df_norm[4] = df[4]
    #iris = df_norm.to_numpy()
    iris = df.to_numpy()
    iris[:, :4] = iris[:, :4]-iris[:, :4].mean(axis=0)
    imax = np.concatenate((iris.max(axis=0)*np.ones((1, 5)),
                           np.abs(iris.min(axis=0))*np.ones((1, 5))), axis=0).max(axis=0)
    iris[:, :4] = iris[:, :4]/imax[:4]
    target = np.zeros((np.shape(iris)[0], 3), dtype=int)
    for x in range(len(replace)):
        target[(lambda y: np.where(iris[:, 4] == y))(x), x] = 1
    return iris, target


def shuffle_data(inputs, target, reseed=False, seed=42):
    if reseed:
        np.random.seed(seed)

    order = [*range(np.shape(inputs)[0])]
    np.random.shuffle(order)
    inputs = inputs[order, :]
    target = target[order, :]

    return inputs, target


def separate_data(inputs, target, validate=False):
    if validate:
        x_in = inputs[::2, 0:4]
        x_out = target[::2]
        v_in = inputs[1::4, 0:4]
        v_out = target[1::4]
        y_in = inputs[3::4, 0:4]
        y_out = target[3::4]
        return x_in, x_out, v_in, v_out, y_in, y_out

    else:
        x_in = inputs[::2, 0:4]
        x_out = target[::2]
        v_in = inputs[1::4, 0:4]
        v_out = target[1::4]
        y_in = inputs[3::4, 0:4]
        y_out = target[3::4]
        x_in_v_in = np.concatenate((x_in, v_in))
        x_out_v_out = np.concatenate((x_out, v_out))

        return x_in_v_in, x_out_v_out, y_in, y_out


def run_mlp(inputs, targets, nRuns=5):
    total_time, total_percent = 0, 0
    for i in range(nRuns):
        x, y = shuffle_data(inputs, targets, True, i)
        x_in, x_out, v_in, v_out, y_in, y_out = separate_data(x, y, True)
        start = time()
        forward = mlp(x_in, x_out,
                      valid_inputs=v_in, valid_targets=v_out, validate=True,
                      outtype="softmax", eta=0.1, nHidden=5,
                      valid_iterations=100)
        predict = forward(y_in)
        total_time += time() - start

        cm = conf_mat(predict, y_out)
        total_percent += np.trace(cm)/np.sum(cm)*100

    return total_time/nRuns, total_percent/nRuns


def run_rbf_pcn(inputs, targets, nRuns=5):
    total_time, total_percent = 0, 0
    for i in range(nRuns):
        x, y = shuffle_data(inputs, targets, True, i)
        x_in, x_out, v_in, v_out, y_in, y_out = separate_data(x, y, True)
        start = time()
        forward = rbf(x_in, x_out,
                      outtype="perceptron",
                      nRbf=5, normalize=True,
                      eta=0.25, iterations=2000)

        predict = forward(y_in)
        total_time += time() - start

        cm = conf_mat(predict, y_out)
        total_percent += np.trace(cm)/np.sum(cm)*100

    return total_time/nRuns, total_percent/nRuns


def run_rbf_wtm(inputs, targets, nRuns=5):
    total_time, total_percent = 0, 0
    for i in range(nRuns):
        x, y = shuffle_data(inputs, targets, True, i)
        x_in, x_out, v_in, v_out, y_in, y_out = separate_data(x, y, True)
        start = time()
        forward = rbf(x_in, x_out,
                      outtype="weight_matrix",
                      nRbf=5, normalize=True)

        predict = forward(y_in)
        total_time += time() - start

        cm = conf_mat(predict, y_out)
        total_percent += np.trace(cm)/np.sum(cm)*100

    return total_time/nRuns, total_percent/nRuns


def run_svm(inputs, targets, nRuns=5):
    total_time, total_percent = 0, 0
    for i in range(nRuns):
        x, y = shuffle_data(inputs, targets, True, i)
        x_in, x_out, y_in, y_out = separate_data(x, y, False)
        for i in range(x_out.shape[0]):
            for j in range(x_out.shape[1]):
                if x_out[i, j] == 0.0:
                    x_out[i, j] = -1.0
        start = time()

        predict = np.zeros((y_out.shape[0], 3))

        classifier0 = svm(x_in, x_out[:, 0], kernel="rbf")
        predict[:, 0] = classifier0(y_in, soft=True).T

        classifier1 = svm(x_in, x_out[:, 1], kernel="rbf")
        predict[:, 1] = classifier1(y_in, soft=True).T

        classifier2 = svm(x_in, x_out[:, 2], kernel="rbf")
        predict[:, 2] = classifier2(y_in, soft=True).T

        total_time += time() - start

        cm = conf_mat(predict, y_out)
        total_percent += np.trace(cm)/np.sum(cm)*100

    return total_time/nRuns, total_percent/nRuns


def run_knn(inputs, targets, nRuns=5):
    total_time, total_percent = 0, 0
    for i in range(nRuns):
        x, y = shuffle_data(inputs, targets, True, i)
        x_in, x_out, y_in, y_out = separate_data(x, y, False)

        start = time()
        predict = knn(3, x_in, x_out, y_in)

        total_time += time() - start

        cm = conf_mat(predict, y_out)
        total_percent += np.trace(cm)/np.sum(cm)*100

    return total_time/nRuns, total_percent/nRuns

def run_dtree(inputs, targets, features, nRuns=5):
    total_time, total_percent = 0, 0
    #classes = [None] * len(targets)
    classes = np.empty((len(targets), 1), dtype=int)
    for i in range(len(targets)):
        if targets[i][0] == 1:
            classes[i] = 0#"Setosa"
        elif targets[i][1] == 1:
            classes[i] = 1#"Versicolour"
        else:
            classes[i] = 2#"Virginica"
    #classes = np.array(classes).reshape(len(targets), 1)
    for i in range(nRuns):
        x, y = shuffle_data(inputs, classes, True, i)
        x_in, x_out, y_in, y_out = separate_data(x, y, False)

        x_out, y_out = x_out[:,0], y_out[:, 0]

        start = time()
        decisionTree = dtree(x_in, x_out, features, maxlevel=1)

        predict = np.empty((len(y_out), 1), dtype=int)
        for idx, val in enumerate(y_in):
            predict[idx] = decisionTree(val)
        total_time += time() - start

        oneHotPredict = np.empty((len(y_out), 3), dtype=int)
        oneHotYout = np.empty((len(y_out), 3), dtype=int)
        for i in range(len(predict)):
            if predict[i] == 0:
                oneHotPredict[i] = [1, 0, 0]
            elif predict[i] == 1:
                oneHotPredict[i] = [0, 1, 0]
            else:
                oneHotPredict[i] = [0, 0, 1]
            if y_out[i] == 0:
                oneHotYout[i] = [1, 0, 0]
            elif y_out[i] == 1:
                oneHotYout[i] = [0, 1, 0]
            else:
                oneHotYout[i] = [0, 0, 1]

        cm = conf_mat(oneHotPredict, oneHotYout)
        total_percent += np.trace(cm)/np.sum(cm)*100

    return total_time/nRuns, total_percent/nRuns


if __name__ == "__main__":
    iris, targets = read_data()

    mlp_time, mlp_percent = run_mlp(iris, targets)

    rbf_pcn_time, rbf_pcn_percent = run_rbf_pcn(iris, targets)

    rbf_wtm_time, rbf_wtm_percent = run_rbf_wtm(iris, targets)

    svm_time, svm_percent = run_svm(iris, targets)

    knn_time, knn_percent = run_knn(iris, targets)

    dtree_time, dtree_percent = run_dtree(iris, targets, \
                                          ["SepalLength", "SepalWidth", \
                                           "PetalLength", "PetalWidth"])

    #=================== Print Results =====================#
    print("\n======== SCORING =========\n")
    print("===== MLP Predictions =====")
    print(f"Percentage Correct: {mlp_percent}")
    print(f"Learning Time: {mlp_time}")
    print("")

    print("===== RBF PCN Predictions =====")
    print(f"Percentage Correct: {rbf_pcn_percent}")
    print(f"Learning Time: {rbf_pcn_time}")
    print("")

    print("===== RBF WTM Predictions =====")
    print(f"Percentage Correct: {rbf_wtm_percent}")
    print(f"Learning Time: {rbf_wtm_time}")
    print("")

    print("===== SVM Predictions =====")
    print(f"Percentage Correct: {svm_percent}")
    print(f"Learning Time: {svm_time}")
    print("")

    print("===== KNN Predictions =====")
    print(f"Percentage Correct: {knn_percent}")
    print(f"Learning Time: {knn_time}")
    print("")

    print("===== DTREE Predictions =====")
    print(f"Percentage Correct: {dtree_percent}")
    print(f"Learning Time: {dtree_time}")
    print("")
