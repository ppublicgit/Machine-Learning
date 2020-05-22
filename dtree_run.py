import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Learning.decision_trees import dtree#ID3, dtreeCART


def sse(actual, predicted):
    ret = 0
    for i in range(len(actual)):
        ret += (actual[i] - predicted[i])**2
    return ret

if __name__ == "__main__":
    filename = os.path.join(os.getcwd(), "Data/party.data")

    df = pd.read_csv(filename)
    features = df.columns[:-1]
    targets = df.iloc[:,-1].values
    data = df.iloc[:, :-1].to_numpy()

    classify = dtree(data, targets, features, treetype="ID3")

    datapoint = ["Urgent", "Yes", "No"]

    print(f"New DTREE ID3: {classify(datapoint)}")

    classify = dtree(data, targets, features, treetype="CART")

    print(f"New DTREE CART: {classify(datapoint)}")

    df = pd.read_csv(os.path.join(os.getcwd(), "Data/machine.data"), header=None)

    features = ["VENDOR", "MODEL", "MYCT", "MMIN", "MMAX",
                "CACH", "CHMIN", "CHMAX"]
    targets = df.iloc[:, -2].values
    data = df.iloc[:, :-2].to_numpy()

    comparison = df.iloc[:,-1].values

    order = [*range(np.shape(data)[0])]
    np.random.shuffle(order)
    inputs = data[order, :]
    target = targets[order]
    comparison = comparison[order]

    x_in = inputs[::2]
    x_out = target[::2]
    v_in = inputs[1::4]
    v_out = target[1::4]
    y_in = inputs[3::4]
    y_out = target[3::4]
    comp_out = comparison[3::4]
    x_in = np.concatenate((x_in, v_in))
    x_out = np.concatenate((x_out, v_out))

    classify = dtree(x_in, x_out, features, outtype="regression")

    predictions = []
    for i in range(len(y_in)):
        predictions.append(classify(y_in[i]))

    print(classify(None))

    print(f"Paper Perf: {sse(y_out, comp_out)}")
    print(f"DTree Perf: {sse(y_out, predictions)}")

    plt.figure()
    plt.plot(y_out, "g*", label="Actual")
    plt.plot(comp_out, "b*", label="Paper")
    plt.plot(predictions, "r*", label="DTree")
    plt.legend()
    plt.show()
