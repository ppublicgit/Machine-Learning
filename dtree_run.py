import os
import numpy as np
import pandas as pd
from Learning.decision_trees import dtree


if __name__ == "__main__":
    filename = os.path.join(os.getcwd(), "Data/party.data")

    df = pd.read_csv(filename)
    features = df.columns
    targets = df.iloc[:,-1].values
    data = df.iloc[:, :-1].to_numpy()

    classify = dtree(data, targets, features)

    datapoint = ["Urgent", "Yes", "No"]

    print(f"New: {classify(datapoint)}")
