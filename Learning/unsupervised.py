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


def som(data, map_shape, **kwargs):

    def forward(inputs):
        if isinstance(inputs, str) and inputs == "getMap":
            return som_map
        activations = np.sum((np.transpose(np.tile(inputs, (x * y, 1))) - weights)**2, axis=0)
        best = np.argmin(activations)
        return best

    def train():
        nonlocal weights, eta_b, eta_neighbour, neighbourSize
        eta_binit, eta_ninit, nSizeInit = eta_b, eta_neighbour, neighbourSize

        for i in range(nIterations):
            for iData in range(nData):
                best = forward(data[iData, :])
                weights[:, best] += eta_b * (data[iData, :] - weights[:, best])

                neighbours = np.where(mapDistances[best, :] <= neighbourSize, 1, 0)
                neighbours[best] = 0
                weights += eta_neighbour  * neighbours * np.transpose((data[iData, :] - weights.T))

            eta_b = eta_binit * np.power(eta_bfinal/eta_binit, float(i)/nIterations)
            eta_neighbour = eta_ninit * np.power(eta_nfinal/eta_ninit, float(i)/nIterations)
            neighbourSize = nSizeInit * np.power(neighbourSizeFinal/nSizeInit, float(i)/nIterations)

    eta_b = kwargs.get("eta_b", 0.3)
    eta_neighbour = kwargs.get("eta_n", 0.1)
    eta_bfinal = kwargs.get("eta_bfinal", 0.03)
    eta_nfinal = kwargs.get("eta_nfinal", 0.01)
    neighbourSize = kwargs.get("neighbourSize", 0.5)
    neighbourSizeFinal = kwargs.get("neighbourSizeFinal", 0.05)
    boundaries = bool(kwargs.get("boundaries", False))
    nIterations = kwargs.get("nIterations", 100)

    x = map_shape[0]
    y = map_shape[1]

    nData = data.shape[0]

    som_map = np.mgrid[0:1:np.complex(0, x), 0:1:np.complex(0, y)]
    som_map = som_map.reshape(2, x * y)

    weights = (np.random.rand(data.shape[1], x * y) - 0.5) * 2

    mapDistances =np.zeros((x * y, x * y))

    if boundaries:
        for i in range(x * y):
            for j in range(i+1, x * y):
                xdist = np.min( (som_map[0,i]-som_map[0,j])**2, \
                                (som_map[0,i]+1+1./x-som_map[0,j])**2, \
                                (som_map[0,i]-1-1./x-som_map[0,j])**2, \
                                (som_map[0,i]-som_map[0,j]+1+1./x)**2, \
                                (som_map[0,i]-som_map[0,j]-1-1./x)**2
                )
                ydist = np.min( (som_map[1,i]-som_map[1,j])**2, \
                                (som_map[1,i]+1+1./y-som_map[1,j])**2, \
                                (som_map[1,i]-1-1./y-som_map[1,j])**2, \
                                (som_map[1,i]-som_map[1,j]+1+1./y)**2, \
                                (som_map[1,i]-som_map[1,j]-1-1./y)**2
                )
                mapDistances[i, j] = np.sqrt(xdist + ydist)
                mapDistances[j, i] = mapDistances[i, j]
    else:
        for i in range(x * y):
            for j in range(i+1, x * y):
                mapDistances[i, j] = np.sqrt( (som_map[0, i] - som_map[0, j])**2 + \
                                              (som_map[1, i] - som_map[1, j])**2
                                              )
                mapDistances[j, i] = mapDistances[i, j]

    train()

    return forward
