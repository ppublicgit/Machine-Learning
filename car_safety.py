import os
import numpy as np
import pandas as pd
from Learning.ensemble_learning import randomForest
from Learning.scoring import conf_mat
from Preprocessing.reformat import one_hot_encode


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(os.getcwd(), "Data/car.data"))

    car = df.to_numpy()

    x_in = car[::2, :-1]
    x_out = car[::2, -1]
    y_in = car[1::2, :-1]
    y_out = car[1::2, -1]

    features = ["buying", "maint", "doors", "persons", "lug_boot", "safety"]

    classify = randomForest(x_in, x_out, features, 5)

    predict = classify(y_in)

    options = ["unacc", "acc", "good", "vgood"]

    predict_check, y_out_check = one_hot_encode(predict, options), one_hot_encode(y_out, options)
    breakpoint()
    conf_mat(predict_check, y_out_check)
