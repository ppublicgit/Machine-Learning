import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from Learning.scoring import conf_mat
from Learning.mlp import mlp
#from Examples.MLCode.Ch4.mlp import mlp as mlp_mars
from Learning.rbf import rbf

def normalize(x, mean, max_val):
    return (x-mean)/max_val

if __name__ == "__main__":
    df = pd.read_csv(os.path.join(os.getcwd(), "Data/iris.data"), header=None)
    replace = [("Iris-setosa", 0), ("Iris-versicolor", 1), ("Iris-virginica", 2)]
    [df.replace(to_replace=replace[i][0], value=replace[i][1], inplace=True) for i in range(len(replace))]
    df = df.astype({4: int})
    df_norm = (df.iloc[:, 0:-1]-df.iloc[:,0:-1].mean())/df.iloc[:, 0:-1].max()
    df_norm[4] = df[4]
    iris = df_norm.to_numpy()
    iris = df.to_numpy()
    iris[:,:4] = iris[:,:4]-iris[:,:4].mean(axis=0)
    imax = np.concatenate((iris.max(axis=0)*np.ones((1,5)),np.abs(iris.min(axis=0))*np.ones((1,5))),axis=0).max(axis=0)
    iris[:,:4] = iris[:,:4]/imax[:4]
    target = np.zeros((np.shape(iris)[0], 3))
    for x in range(len(replace)):
        target[(lambda y : np.where(iris[:,4]==y))(x), x] = 1

    order = [*range(np.shape(iris)[0])]
    np.random.shuffle(order)
    iris = iris[order, :]
    target = target[order, :]

    x_in = iris[::2,0:4]
    x_out = target[::2]
    v_in = iris[1::4, 0:4]
    v_out = target[1::4]
    y_in = iris[3::4, 0:4]
    y_out = target[3::4]

    #mm = mlp_mars(x_in, x_out, 5, outtype="softmax")
    #mm.earlystopping(x_in, x_out, v_in, v_out, 0.1)
    #mm.confmat(y_in, y_out)

    forward_mlp = mlp(x_in, x_out,
                  valid_inputs=v_in, valid_targets=v_out, validate=True,
                  outtype="softmax", eta=0.1, nHidden=5,
                  valid_iterations=100)

    predict_mlp = forward_mlp(y_in)

    forward_rbf_pcn = rbf(x_in, x_out,
                  outtype="perceptron",
                  nRbf=5, normalize=True,
                  eta=0.25, iterations=2000)

    predict_rbf_pcn = forward_rbf_pcn(y_in)

    forward_rbf_wtm = rbf(x_in, x_out,
                  outtype="weight_matrix",
                  nRbf=5, normalize=True)

    predict_rbf_wtm = forward_rbf_wtm(y_in)

    print("===== MLP Predictions =====")
    conf_mat(predict_mlp, y_out)

    print("===== RBF PCN Predictions =====")
    conf_mat(predict_rbf_pcn, y_out)

    print("===== RBF WTM Predictions =====")
    conf_mat(predict_rbf_wtm , y_out)
