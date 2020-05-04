import numpy as np
from perceptron import Perceptron
from linreg import LinearRegression
from mlp import MultiLayerPerceptron
from TestInputs import *

if __name__ == "__main__":

    print("===Perceptron Test===")
    pcn = Perceptron(XOR3D().inputs, XOR3D().targets)
    pcn.train(0.25, 10)
    pcn.print_outputs()
    pcn.conf_mat()
    print("")

    print("===Linear Regression Test===")
    lr = LinearRegression(XOR2D().inputs, XOR2D().targets)
    lr.train()
    print(lr.get_train_output())
    print("")

    print("===Multi Layer Perceptron Test===")
    mlp = MultiLayerPerceptron(XOR3D().inputs, XOR3D().targets,2)
    mlp.train(0.25, 1001)
    print(mlp.get_train_outputs())
    mlp.conf_mat()
    print("")
