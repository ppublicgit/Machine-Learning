import numpy as np
import random
from .decision_trees import dtree


def randomForest(data, targets, features, nTrees, **kwargs):

    def classify(randForest, data):
        decisions = []
        # Majority voting
        for datapoint in data:
            outputs = []
            for tree in randForest:
                out = tree(datapoint)
                outputs.append(out)
                if out is None:
                    raise ValueError("None out for rnad forest")
                    outputs.append(out)
            # List the possible outputs
            if outtype == "classification":
                predictions = [*set(outputs)]
                frequency = np.zeros(len(predictions))
                for output in outputs:
                    frequency[predictions.index(output)] += 1
                decisions.append(predictions[frequency.argmax()])
            else:
                decisions.append(np.median(outputs))
        return decisions

    def genForest(data, targets, features, nTrees, nSamples, **kwargs):
        treeForest = []
        for i in range(nTrees):
            bootstrapData, bootstrapTargets = bootstrap(data, targets, nSamples)
            treeForest.append(dtree(bootstrapData, bootstrapTargets, features, **kwargs))
        return treeForest

    def bootstrap(data, targets, nSamples):
        nData = data.shape[0]
        if replacement:
            samplePoints = np.random.randint(0,nData, (nData))
        else:
            samplePoints = random.sample(range(nData), nSamples)
        sampleData = []
        sampleTarget = []
        for j in range(nSamples):
            sampleData.append(data[samplePoints[j], :])
            sampleTarget.append(targets[samplePoints[j]])
        return np.array(sampleData), np.array(sampleTarget)

    nSamples = kwargs.pop("nSamples", 0)
    replacement = bool(kwargs.pop("replacement", False))
    outtype = kwargs.get("outtype", "classification")

    if outtype not in ["classification", "regression"]:
        raise ValueError("Invalid outtype for random forest")

    if nSamples <= 0 and replacement:
        nSamples = len(data)
    elif nSamples <= 0:
        nSamples = len(data)//2
    elif nSample > len(data):
        nSamples = len(data)

    randForest = genForest(data, targets, features, nTrees, nSamples, **kwargs)

    return lambda x: classify(randForest, x)
