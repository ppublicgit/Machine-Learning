import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Learning.ensemble_learning import randomForest


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

    classify = randomForest(data, targets, features, 5, maxlevel=2)

    datapoint = [["Urgent", "Yes", "No"]]

    print(f"Rand Forest 5 Trees 2 Maxlevel: {classify(datapoint)[0]}")

    classify = randomForest(data, targets, features, 10, maxlevel=1)

    print(f"Rand Forest 10 Trees 1 Maxlevel: {classify(datapoint)[0]}")

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

    classify = randomForest(x_in, x_out, features, 10, maxlevel=5, outtype="regression")

    predictions = classify(y_in)

    print(f"Paper Perf: {sse(y_out, comp_out)}")
    print(f"DTree Perf: {sse(y_out, predictions)}")

    plt.figure()
    plt.plot(y_out, "g*", label="Actual")
    plt.plot(comp_out, "b*", label="Paper")
    plt.plot(predictions, "r*", label="DTree")
    plt.legend()
    plt.show()
