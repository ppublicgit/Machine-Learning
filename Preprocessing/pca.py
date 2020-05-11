# Code is based off of Stephen Marsland's code
# for Machine Learning Second Edition Book

import numpy as np


def pca(data, nRedDim=0, normalize=True):

    m = np.mean(data, axis=0)
    data_c = data-m

    cov = np.cov(data.T)

    evals, evecs = np.linalg.eig(C)
    indices = np.argsort(evals)
    indices = indices[::-1]
    evecs = evecs[:, indices]
    evals = evals[:, indices]

    if nRedDim > 0:
        evecs = [:, :nRedDim]

    if normalize:
        for i in range(evecs.shape[1]):
            evecs[:, i] /= np.linalg.norm(evecs[:, i]) * np.sqrt(evals[i])

    x = np.dot(evecs.T, data.T)
    y = (np.dot(evecs, x)).T + m

    return x, y, evals, evecs


def kernelmatrix(data, kernel_type, param=np.array([3, 2])):
    k_matrix = None
    if kernel_type == "linear":
        k_matrix = np.dot(data, data.T)
    elif kernel_type == "gaussian":
        K = np.zeros((data.shape[0], data.shape[0]))
        for i in range(data.shape[0]):
            for j in range(i+1, data.shape[0]):
                K[i, j] = np.sum((data[i, :] - data[j, :])**2)
                K[j, i] = K[i, j]
        k_matrix = np.exp(-K**2/(2*param[0]**2))
    elif kernel_type == "polynomial":
        k_matrix = (np.dot(data, data.T) + param[0])**param[1]
    return k_matrix


def kernelpca(data, kernel_type, nRedDim):

    nData = data.shape[0]

    K = kernelmatrix(data, kernel_type)

    # Compute the transformed data
    D = np.sum(K, axis=0)/nData
    E = np.sum(D)/nData
    J = np.ones((nData, 1))*D
    K = K - J - J.T + E*np.ones((nData, nData))

    # Perform the dimensionality reduction
    evals, evecs = np.linalg.eig(K)
    indices = np.argsort(evals)
    indices = indices[::-1]
    evecs = evecs[:, indices[:nRedDim]]
    evals = evals[indices[:nRedDim]]

    sqrtE = np.zeros((len(evals), len(evals)))
    for i in range(len(evals)):
        sqrtE[i, i] = np.sqrt(evals[i])

    newData = np.transpose(np.dot(sqrtE, evecs.T))

    return newData


def factor_analysis(y, nRedDim):
    # shapes of data
    Ndata = np.shape(y)[0]
    N = np.shape(y)[1]
    # zero mean the data
    y = y-y.mean(axis=0)

    C = np.cov(np.transpose(y))
    Cd = C.diagonal()
    Psi = Cd
    scaling = np.linalg.det(C)**(1./N)

    W = np.random.normal(0, np.sqrt(scaling/nRedDim), (N, nRedDim))

    nits = 1000
    oldL = -np.inf

    for i in range(nits):

        # E-step
        A = np.dot(W, np.transpose(W)) + np.diag(Psi)
        logA = np.log(np.abs(np.linalg.det(A)))
        A = np.linalg.inv(A)

        WA = np.dot(np.transpose(W), A)
        WAC = np.dot(WA, C)
        Exx = np.eye(nRedDim) - np.dot(WA, W) + np.dot(WAC, np.transpose(WA))

        # M-step
        W = np.dot(np.transpose(WAC), np.linalg.inv(Exx))
        Psi = Cd - (np.dot(W, WAC)).diagonal()
        #Sigma1 = (dot(transpose(y),y) - dot(W,WAC)).diagonal()/Ndata

        tAC = (A*np.transpose(C)).sum()

        L = -N/2*np.log(2.*np.pi) - 0.5*logA - 0.5*tAC
        if (L-oldL) < (1e-4):
            print "Stop", i
            break
        print L
        oldL = L
    A = np.linalg.inv(np.dot(W, np.transpose(W))+np.diag(Psi))
    Ex = np.dot(np.transpose(A), W)

    return np.dot(y, Ex)


def lle(data, nRedDim=2, K=12):

    ndata = np.shape(data)[0]
    ndim = np.shape(data)[1]
    d = np.zeros((ndata, ndata), dtype=float)

    # Inefficient -- not matrices
    for i in range(ndata):
        for j in range(i+1, ndata):
            for k in range(ndim):
                d[i, j] += (data[i, k] - data[j, k])**2
            d[i, j] = np.sqrt(d[i, j])
            d[j, i] = d[i, j]

    indices = d.argsort(axis=1)
    neighbours = indices[:, 1:K+1]

    W = np.zeros((K, ndata), dtype=float)

    for i in range(ndata):
        Z = data[neighbours[i, :], :] - np.kron(np.ones((K, 1)), data[i, :])
        C = np.dot(Z, np.transpose(Z))
        C = C+np.identity(K)*1e-3*np.trace(C)
        W[:, i] = np.transpose(np.linalg.solve(C, np.ones((K, 1))))
        W[:, i] = W[:, i]/np.sum(W[:, i])

    M = np.eye(ndata, dtype=float)
    for i in range(ndata):
        w = np.transpose(np.ones((1, np.shape(W)[0]))*np.transpose(W[:, i]))
        j = neighbours[i, :]
        # print shape(w), np.shape(np.dot(w,np.transpose(w))), np.shape(M[i,j])
        ww = np.dot(w, np.transpose(w))
        for k in range(K):
            M[i, j[k]] -= w[k]
            M[j[k], i] -= w[k]
            for l in range(K):
                M[j[k], j[l]] += ww[k, l]

    evals, evecs = np.linalg.eig(M)
    ind = np.argsort(evals)
    y = evecs[:, ind[1:nRedDim+1]]*np.sqrt(ndata)
    return evals, evecs, y


def isomap(data, newdim=2, K=12, labels=None):

    ndata = np.shape(data)[0]
    ndim = np.shape(data)[1]
    d = np.zeros((ndata, ndata), dtype=float)

    # Compute the distance matrix
    # Inefficient -- not matrices
    for i in range(ndata):
        for j in range(i+1, ndata):
            for k in range(ndim):
                d[i, j] += (data[i, k] - data[j, k])**2
            d[i, j] = np.sqrt(d[i, j])
            d[j, i] = d[i, j]

    # K-nearest neighbours
    indices = d.argsort()
    #notneighbours = indices[:,K+1:]
    neighbours = indices[:, :K+1]
    # Alternative: epsilon
    # epsilon = 0.1
    #neighbours = where(d<=epsilon)
    #notneighbours = where(d>epsilon)

    h = np.ones((ndata, ndata), dtype=float)*np.inf
    for i in range(ndata):
        h[i, neighbours[i, :]] = d[i, neighbours[i, :]]

    # Compute the full distance matrix over all paths
    print "Floyd's algorithm"
    for k in range(ndata):
        for i in range(ndata):
            for j in range(ndata):
                if h[i, j] > h[i, k] + h[k, j]:
                    h[i, j] = h[i, k] + h[k, j]

#	print "Dijkstra's algorithm"
#	q = h.copy()
#	for i in range(ndata):
#		for j in range(ndata):
#			k = np.argmin(q[i,:])
#			while not(np.isinf(q[i,k])):
#				q[i,k] = np.inf
#				for l in neighbours[k,:]:
#					possible = h[i,l] + h[l,k]
#					if possible < h[i,k]:
#						h[i,k] = possible
#				k = np.argmin(q[i,:])
#	print "Comnlete"

    # remove lines full of infs
    x = np.isinf(h[:, 0]).nonzero()
    if np.size(x) > 0:
        print x
        if x[0][0] > 0:
            new = h[0:x[0][0], :]
            newlabels = labels[0:x[0][0]]
            start = 1
        else:
            new = h[x[0][0]+1, :]
            newlabels = labels[x[0][0]+1]
            start = 2
        for i in range(start, size(x)):
            new = np.concatenate((new, h[x[0][i-1]+1:x[0][i], :]), axis=0)
            newlabels = np.concatenate(
                (newlabels, labels[x[0][i-1]+1:x[0][i]]), axis=0)
        new = np.concatenate((new, h[x[0][i]+1:, :]), axis=0)
        newlabels = np.concatenate((newlabels, labels[x[0][i]+1:]), axis=0)

        new2 = new[:, 0:x[0][0]]
        if x[0][0] > 0:
            new2 = new[:, 0:x[0][0]]
            start = 1
        else:
            new2 = new[:, x[0][0]+1]
            start = 2
        for i in range(start, size(x)):
            new2 = np.concatenate((new2, new[:, x[0][i-1]+1:x[0][i]]), axis=1)
        new2 = np.concatenate((new2, new[:, x[0][i]+1:]), axis=1)

        g = new2.copy()
        ndata = ndata - size(x)
    else:
        g = h.copy()
        newlabels = labels

    # Map computations, following by the dimensionality reduction
    M = -0.5*(g**2 - np.transpose(np.sum(g*g, axis=0) * np.ones((ndata, 1))/ndata) -
              np.ones((ndata, 1)) * np.sum(g*g, axis=0)/ndata + np.sum(np.sum(g*g))/ndata**2)

    eval, evec = np.linalg.eig(M)
    eval = np.real(eval)
    ind = np.argsort(eval)
    eval = np.real(np.diag(eval[ind[-1::-1]]))
    evec = evec[:, ind[-1::-1]]
    y = np.real(np.dot(evec, np.transpose((np.sqrt(eval)))))
    print np.shape(y)
    print np.shape(eval), np.shape(evec)
    return y, newlabels
