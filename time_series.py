import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime

from Learning.mlp import mlp


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(os.getcwd(), "Data/PNOz.dat"),
                     header=None, names=["Year", "Day", "Ozone", "SO2"],
                     skipinitialspace=True,
                     sep=' ')

    df["Ozone"] = (df["Ozone"]-df["Ozone"].mean())/df["Ozone"].max()

    t = 2
    k = 3

    PNoz = df.to_numpy()

    lastPoint = np.shape(PNoz)[0]-t*(k+1)
    inputs = np.zeros((lastPoint,k))
    targets = np.zeros((lastPoint,1))
    for i in range(lastPoint):
        inputs[i,:] = PNoz[i:i+t*k:t,2]
        targets[i] = PNoz[i+t*(k+1),2]

    x_in = inputs[:-400:2, :]
    x_out = targets[:-400:2]
    v_in = inputs[1:-400:2, :]
    v_out = inputs[1:-400:2]
    y_in = inputs[-400:, :]
    y_out = targets[-400:]

    # Randomly order the data
    change = [*range(np.shape(inputs)[0])]
    np.random.shuffle(change)
    inputs = inputs[change,:]
    targets = targets[change,:]

    forward = mlp(x_in, x_out, validate=True,
                  valid_inputs=v_in, valid_targets=v_out,
                  outtype="linear",
                  valid_iterations=1000,
                  nHidden=4,
                  eta=0.25)

    if len(y_in.shape) == 1:
        y_in = y_in.reshape(y_in.shape[0], 1)
    predict = forward(y_in)

    plt.figure(1)
    plt.plot(np.arange(np.shape(y_out)[0]), y_out, "r*", label="Data")
    plt.plot(np.arange(np.shape(y_out)[0]), predict, "*g", label="Predict")
    plt.legend()
    plt.show()

    #input("Press Enter to exit...")
