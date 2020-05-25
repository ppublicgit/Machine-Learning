import numpy as np


def one_hot_encode(oneDarray, options):
    oneHotEncoded = np.zeros((len(oneDarray), len(options)))
    for i in range(len(oneDarray)):
        for j, option in enumerate(options):
            if oneDarray[i] == option:
                oneHotEncoded[i][j] = 1

    return oneHotEncoded
