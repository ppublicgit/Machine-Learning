import numpy as np
from copy import deepcopy


def kmeans(data, kClusters, **kwargs):

    def classify(data):
        nData = data.shape[0]
        distances = calc_distances(data, centers)
        return distances.argmin(axis=0).reshape(nData, 1)

    def calc_distances(data, cents):
        distances = np.zeros((kClusters, nData))
        if measure == "euclidean":
            for i in range(nData):
                for j in range(kClusters):
                    distances[j, :] = np.sum((data - centers[j, :])**2, axis=1)
        return distances

    def updateCenters(cluster):
        if average == "mean":
            return np.sum(data * cluster, axis=0) / np.sum(cluster)
        elif average == "median":
            return np.median(data * cluster, axis=0)

    def train():
        nonlocal centers, prevCenters

        iteration = 0
        while np.sum(np.sum(prevCenters-centers)) >= cutoffChange and \
              iteration < maxIterations:
            prevCenters = deepcopy(centers)
            distances = calc_distances(data, centers)
            cluster = distances.argmin(axis=0).reshape(nData, 1)
            for j in range(kClusters):
                thisCluster = np.where(cluster==j, 1, 0)
                if sum(thisCluster) > 0:
                    centers[j, :] = updateCenters(thisCluster)
            iteration += 1
        return

    measure = kwargs.get("measure", "euclidean")
    average = kwargs.get("average", "mean")
    cutoffChange = kwargs.get("cutoff", 0)
    maxIterations = kwargs.get("maxIterations", 10)

    if measure not in ["euclidean"]:
        raise ValueError(f"Measure {measure} is an invalid option.")
    if average not in ["mean", "median"]:
        raise ValueError(f"Average {average} is an invalid option.")

    nData = data.shape[0]
    nFeatures = data.shape[1]

    minima = data.min(axis=0)
    maxima = data.max(axis=0)

    centers = np.random.rand(kClusters, nFeatures) * (maxima-minima) + minima
    prevCenters = np.random.rand(kClusters, nFeatures) * (maxima-minima) + minima

    train()

    return lambda x : classify(x)


def kmeansnet(inputs, kClusters, **kwargs):

    def classify(data):
        best = np.zeros((data.shape[0], 1), dtype=int)
        for i in range(data.shape[0]):
            best[i] = np.argmax(np.sum(weights * data[i:i+1, :].T, axis=0))
        return best

    def train():
        nonlocal data, weights
        normalisers = np.sqrt(np.sum(data**2, axis=1)).reshape(1, data.shape[0])
        data = np.transpose(data.T / normalisers)

        for i in range(nEpochs):
            for j in range(data.shape[0]):
                winner = np.argmax(np.sum(weights * data[j:j+1, :].T, axis=0))
                weights[:, winner] += eta * data[j, :] - weights[:, winner]
        return

    nEpochs = kwargs.get("nEpochs", 1000)
    eta = kwargs.get("eta", 0.25)

    data = deepcopy(inputs)

    weights = np.random.rand(data.shape[1], kClusters)

    train()

    return classify
