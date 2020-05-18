import numpy as np

#
# def GMM(y):
#    """ Fits two Gaussians to data using the EM algorithm
#    """
#
#    # Initialisation
#    mu1 = y[np.random.randint(0, N-1, 1)]
#    mu2 = y[np.random.randint(0, N-1, 1)]
#    s1 = np.sum((y-np.mean(y))**2)/N
#    s2 = s1
#    pi = 0.5
#
#    # EM loop
#    count = 0
#    gamma = 1.*np.zeros(N)
#    nits = 20
#
#    ll = 1.*np.zeros(nits)
#
#    while count < nits:
#        count += 1
#
#        # E-step
#        for i in range(N):
#            gamma[i] = pi*np.exp(-(y[i]-mu1)**2/(2*s1)) / (
#                pi * np.exp(-(y[i]-mu1)**2/(2*s1)) + (1-pi) * np.exp(-(y[i]-mu2)**2/2*s2))
#
#        # M-step
#        mu1 = np.sum((1-gamma)*y)/np.sum(1-gamma)
#        mu2 = np.sum(gamma*y)/np.sum(gamma)
#        s1 = np.sum((1-gamma)*(y-mu1)**2)/np.sum(1-gamma)
#        s2 = np.sum(gamma*(y-mu2)**2)/np.sum(gamma)
#        pi = np.sum(gamma)/N
#
#        ll[count-1] = np.sum(np.log(pi*np.exp(-(y[i]-mu1) **
#                                              2/(2*s1)) + (1-pi)*np.exp(-(y[i]-mu2)**2/(2*s2))))
#
#    x = np.arange(-2, 8.5, 0.1)
#    y = 35*pi*np.exp(-(x-mu1)**2/(2*s1)) + 35*(1-pi)*np.exp(-(x-mu2)**2/(2*s2))


def knn(kNeighbors, data, data_class, inputs, metric="L2"):
    nInputs = inputs.shape[0]
    closest = np.zeros((nInputs, data_class.shape[1]))

    for n in range(nInputs):
        if metric == "L2":
            distances = np.sum((data-inputs[n, :])**2, axis=1)
        elif metric == "L1":
            distances = np.sum((data-inputs[n, :]), axis=1)
        else:
            raise ValueError("Invalid metric specified for call to knn")

        indices = np.argsort(distances, axis=0)


        classes = np.unique(data_class[indices[:kNeighbors]], axis=0)
        if classes.shape[0] == 1:
            closest[n] = classes
        else:
            counts = np.zeros(data_class.shape[1])
            for i in range(kNeighbors):
                for j in range(data_class.shape[1]):
                    counts[j] += data_class[indices[i], j]
            max_neighbor = np.argmax(counts)
            if closest.shape[1] == 1:
                closest[n] = max_neighbor
            else:
                max_neighbor_array = np.zeros(data_class.shape[1])
                max_neighbor_array[max_neighbor] = 1
                closest[n] = max_neighbor_array
    return closest


def knn_regression(k, data, testpoints, kernel):

    outputs = np.zeros(len(testpoints))

    for i in range(len(testpoints)):
        distances = (data[:, 0]-testpoints[i])
        if kernel == 'NN':
            indices = np.argsort(distances**2, axis=0)
            outputs[i] = 1./k * np.sum(data[indices[:k], 1])
        elif kernel == 'Epan':
            Klambda = 0.75*(1 - distances**2/k**2)
            where = (np.abs(distances) < k)
            outputs[i] = np.sum(Klambda*where*data[:, 1])/np.sum(Klambda*where)
        elif kernel == 'Tricube':
            Klambda = (1 - np.abs((distances/k)**3)**3)
            where = (np.abs(distances) < k)
            outputs[i] = np.sum(Klambda*where*data[:, 1])/np.sum(Klambda*where)
        else:
            print('Unknown kernel')
    return outputs


class node:
    # A passive class to hold the nodes
    pass


def makeKDtree(points, depth):
    if np.shape(points)[0] < 1:
                # Have reached an empty leaf
        return None
    elif np.shape(points)[0] < 2:
        # Have reached a proper leaf
        newNode = node()
        newNode.point = points[0, :]
        newNode.left = None
        newNode.right = None
        return newNode
    else:
        # Pick next axis to split on
        whichAxis = np.mod(depth, np.shape(points)[1])

        # Find the median point
        indices = np.argsort(points[:, whichAxis])
        points = points[indices, :]
        median = np.ceil(float(np.shape(points)[0]-1)/2)

        # Separate the remaining points
        goLeft = points[:median, :]
        goRight = points[median+1:, :]

        # Make a new branching node and recurse
        newNode = node()
        newNode.point = points[median, :]
        newNode.left = makeKDtree(goLeft, depth+1)
        newNode.right = makeKDtree(goRight, depth+1)
        return newNode


def returnNearest(tree, point, depth, metric="L2"):
    if tree.left is None:
        # Have reached a leaf
        if metric == "L2":
            distance = np.sum((tree.point-point)**2)
        elif metric == "L1":
            distance = np.sum(tree.point - point)
        else:
            raise ValueError("Invalid metric specified for call to returnNearest")
        return tree.point, distance, 0
    else:
        # Pick next axis to split on
        whichAxis = np.mod(depth, np.shape(point)[0])

        # Recurse down the tree
        if point[whichAxis] < tree.point[whichAxis]:
            bestGuess, distance, height = returnNearest(
                tree.left, point, depth+1)
        else:
            bestGuess, distance, height = returnNearest(
                tree.right, point, depth+1)

        if height <= 2:
            # Check the sibling
            if point[whichAxis] < tree.point[whichAxis]:
                bestGuess2, distance2, height2 = returnNearest(
                    tree.right, point, depth+1)
            else:
                bestGuess2, distance2, height2 = returnNearest(
                    tree.left, point, depth+1)

            # Check this node
            distance3 = np.sum((tree.point-point)**2)
            if (distance3 < distance2):
                distance2 = distance3
                bestGuess2 = tree.point

            if (distance2 < distance):
                distance = distance2
                bestGuess = bestGuess2
        return bestGuess, distance, height+1
