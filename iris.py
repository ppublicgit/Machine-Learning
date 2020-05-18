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

    order = [*range(np.shape(iris)[0])]
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
                      valid_inputs=v_in, valid_targets=v_out, train_type="validate",
                      outtype="softmax", eta=0.1, nHidden=5,
                      valid_iterations=100)
        predict = forward(y_in)
        total_time += time() - start

        cm = conf_mat(predict, y_out)
        total_percent += np.trace(cm)/np.sum(cm)*100

    return total_time/nRuns, total_percent/nRuns

def run_mlp_conj_grad(inputs, targets, nRuns=5):
    total_time, total_percent = 0, 0
    for i in range(nRuns):
        x, y = shuffle_data(inputs, targets, True, i)
        x_in, x_out, y_in, y_out = separate_data(x, y, False)
        start = time()
        forward = mlp(x_in, x_out,
                      train_type="conj_grad",
                      outtype="softmax", nHidden=5)
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


if __name__ == "__main__":
    iris, targets = read_data()

#    mlp_time, mlp_percent = run_mlp(iris, targets)
#
    mlp_conj_grad_time, mlp_conj_grad_percent = run_mlp_conj_grad(iris, targets)
#
#    rbf_pcn_time, rbf_pcn_percent = run_rbf_pcn(iris, targets)
#
#    rbf_wtm_time, rbf_wtm_percent = run_rbf_wtm(iris, targets)
#
#    svm_time, svm_percent = run_svm(iris, targets)
#
#    knn_time, knn_percent = run_knn(iris, targets)
#
#    #=================== Print Results =====================#
#    print("\n======== SCORING =========\n")
#    print("===== MLP Predictions =====")
#    print(f"Percentage Correct: {mlp_percent}")
#    print(f"Learning Time: {mlp_time}")
#    print("")
#
    print("===== MLP Conj. Grad. Predictions =====")
    print(f"Percentage Correct: {mlp_conj_grad_percent}")
    print(f"Learning Time: {mlp_conj_grad_time}")
    print("")
#
#    print("===== RBF PCN Predictions =====")
#    print(f"Percentage Correct: {rbf_pcn_percent}")
#    print(f"Learning Time: {rbf_pcn_time}")
#    print("")
#
#    print("===== RBF WTM Predictions =====")
#    print(f"Percentage Correct: {rbf_wtm_percent}")
#    print(f"Learning Time: {rbf_wtm_time}")
#    print("")
#
#    print("===== SVM Predictions =====")
#    print(f"Percentage Correct: {svm_percent}")
#    print(f"Learning Time: {svm_time}")
#    print("")
#
#    print("===== KNN Predictions =====")
#    print(f"Percentage Correct: {knn_percent}")
#    print(f"Learning Time: {knn_time}")
#    print("")
