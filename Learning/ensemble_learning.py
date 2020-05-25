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
                if out is not None:
                    outputs.append(out)
            # List the possible outputs
            if outtype == "classification":
                predictions = [*set(outputs)]
                if len(predictions) == 0:
                    raise ValueError("no predictions made")
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
    forest = kwargs.pop("nForest", len(features)//2)

    if outtype not in ["classification", "regression"]:
        raise ValueError("Invalid outtype for random forest")

    if nSamples <= 0 and replacement:
        nSamples = len(data)
    elif nSamples <= 0:
        nSamples = len(data)//2
    elif nSample > len(data):
        nSamples = len(data)

    randForest = genForest(data, targets, features, nTrees, nSamples, forest=forest, **kwargs)

    return lambda x: classify(randForest, x)


def boosting(data, targets, features, **kwargs):

    def classify(boosted_cls, data):
        return boosted_cls(data)

    def update_weights(targets, outputs, weights):
        wrong_indices = []
        for i in range(len(targets)):
            if targets[i] != outputs[i]:
                wrong_indices.append(i)
        total_wrong = len(wrong_indices)
        if total_wrong == 0:
            return weights, True
        alpha = calc_alpha(wrong_indices, weights)
        for wrg_idx in wrong_indices:
            weights[wrg_idx] *= alpha
        weights = weights/np.sum(weights)
        if weight_check(weights):
            return weights, True
        else:
            return weights, False

    def calc_alpha(wrong_indices, weights):
        error = 0
        for wrong_idx in wrong_indices:
            error += sum(weights[wrong_idx])
        return (1-error)/error

    def weight_check(weights):
        for wt in weights:
            if sum(wt) >= 0.5:
                return True
        return False

    def boost(data, targets, features, **kwargs):
        nonlocal weights
        for i in range(numBoosts):
            tree = dtree(data, targets, features, weights=weights, **kwargs)
            outputs = classify(tree, data)
            weights, brk = update_weights(targets, outputs, weights)
            if brk:
                break
        breakpoint()
        return tree

    outtype = kwargs.get("outtype", "classification")
    numBoosts = kwargs.pop("num_boosts", 20)

    classifier = kwargs.pop("classifier", "tree")
    if classifier != "tree":
        raise ValueError("Only tree classifier currently supported for boosting")

    if outtype not in ["classification"]:
        raise ValueError("Outtype for boosing must be classification")

    weights = np.ones((data.shape), dtype=float)/data.shape[0]

    boosted = boost(data, targets, features, **kwargs)

    return lambda x: classify(boosted, x)
