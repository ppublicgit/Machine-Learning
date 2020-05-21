import numpy as np

def floatGE(left, right, epsilon=1e-5):
    if left + epsilon >= right:
        return True
    else:
        return False

def floatLT(left, right, epsilon=1e-5):
    if left < right + epsilon:
        return True
    else:
        return False

class DecisionTree:
    def __init__(self, key, subtrees=[]):
        if isinstance(subtrees, list):
            self.subtrees = subtrees
        else:
            self.subtrees = [subtrees]
        self.key = key
        self._spacer = 2
        self._print_level = 0

    def __str__(self):
        return self._print_help()

    def _print_help(self):
        spacer = self._spacer*self._print_level*" "
        if self.key is None:
            return ""
        elif self.subtrees == []:
            return f"{spacer}{self.key}"
        else:
            printKey = f"{spacer}{self.key}\n"
            for subtree in self.subtrees:
                subtree._print_level += 1 + self._print_level
            printSubtrees = "\n".join([subtree.__str__() for subtree in self.subtrees])
            for subtree in self.subtrees:
                subtree._print_level -= 1 + self._print_level
            return printKey + printSubtrees

    def isLeaf(self):
        if self.subtrees == []:
            return True
        return False

    def _index(self, path, discrete=True):
        if discrete:
            for idx, val in enumerate(self.subtrees):
                if val.key == path:
                    return idx
        else:
            for idx, val in enumerate(self.subtrees):
                if val.key[0] == "<" and floatLT(path, float(val.key[1:])):
                    return idx
                elif val.key[0] == ">" and floatGE(path, float(val.key[2:])):
                    return idx
        raise ValueError(f"Key {key} not in subtrees")

    def keys(self):
        return [st.key for st in self.subtrees]

    def subtree(self, path):
        if isinstance(path, str):
            if path not in self.keys():
                return None
            return self.subtrees[self._index(path, True)].subtrees[0]
        else:
            return self.subtrees[self._index(path, False)].subtrees[0]

def dtree(data, targets, features, **kwargs):
    def classify(tree, datapoint):
        if tree is None:
            return None
        elif tree.isLeaf():
            return tree.key
        else:
            for idx, feature in enumerate(featureNames):
                if feature == tree.key:
                    break
            subtree = tree.subtree(datapoint[idx])
            return classify(subtree, datapoint)

    def count(lst, obj):
        count = 0
        for item in lst:
            if item == obj:
                count += 1
        return count

    def make_tree(data, classes, featureNames, maxlevel=-1, level=0, forest=0):
        """ The main function, which recursively constructs the tree"""
        if len(data) == 0:
            return DecisionTree(None, [])
        nData = len(data)
        nFeatures = len(data[0])
        newClasses = [*set(classes)]
        frequency = np.zeros(len(newClasses))
        totalEntropy, totalGini = 0, 0
        for i in range(len(newClasses)):
            frequency[i] = count(classes, newClasses[i])#classes.count(newClasses[i])
            totalEntropy += calc_entropy(float(frequency[i])/nData)
            totalGini += (float(frequency[i])/nData)**2
        totalGini = 1 - totalGini

        if nData == 0 or nFeatures == 0 or (maxlevel >= 0 and level > maxlevel):
            return DecisionTree(newClasses[np.argmax(frequency)], [])
        elif count(classes, classes[0]) == nData:
            return DecisionTree(classes[0], [])
        else:
            # Choose which feature is best
            gain, ggain, cutoffPoints = np.zeros(nFeatures), np.zeros(nFeatures), np.zeros(nFeatures)
            featureSet = [*range(nFeatures)]
            if forest != 0:
                np.random.shuffle(featureSet)
                featureSet = featureSet[0:forest]
            for featureIndex in featureSet:
                if isinstance(data[0][featureIndex], str):
                    cutoffPoint = None
                else:
                    cutoffPoint = get_cutoff_point(data, classes, featureIndex)
                g, gg = calc_info_gain(data, classes, featureIndex, cutoffPoint)
                gain[featureIndex] = totalEntropy - g
                ggain[featureIndex] = totalGini - gg
                cutoffPoints[featureIndex] = cutoffPoint
            bestFeature = np.argmax(gain)
            if isinstance(data[0][featureIndex], str):
                values = [*set([datapoint[bestFeature] for datapoint in data])]
            else:
                values = [">=" + str(cutoffPoints[bestFeature]), "<" + str(cutoffPoints[bestFeature])]
            tree = DecisionTree(featureNames[bestFeature], [])
            # List the values that bestFeature can take
            for value in values:
                # Find the datapoints with each feature value
                newData, newClasses, newNames  = [], [], []
                for index, datapoint in enumerate(data):
                    if matches(datapoint[bestFeature], value) and value[0] != "<":
                        newDatapoint, newNames = extract_data(bestFeature, datapoint, featureNames)
                        newData.append(newDatapoint)
                        newClasses.append(classes[index])
                    elif matches(datapoint[bestFeature], value) and value[0] == "<":
                        newData.append(datapoint)
                        newClasses.append(classes[index])
                        newNames = featureNames
                # Now recurse to the next level
                subtree = make_tree(
                    newData, newClasses, newNames, maxlevel, level+1, forest)
                # And on returning, add the subtree on to the tree
                tree.subtrees.append(DecisionTree(value, subtree))
            return tree

    def matches(datapointVal, checkVal):

        if isinstance(datapointVal, str):
            return datapointVal == checkVal
        elif checkVal[0] == "<":
            return floatLT(datapointVal, float(checkVal[1:]))
        else:
            return floatGE(datapointVal, float(checkVal[2:]))

    def extract_data(featureIndex, datapoint, featureNames):
        if featureIndex == 0:
            newdatapoint = datapoint[1:]
            newNames = featureNames[1:]
        elif featureIndex == len(datapoint):
            newdatapoint = datapoint[:-1]
            newNames = featureNames[:-1]
        else:
            newdatapoint = np.concatenate([datapoint[:featureIndex], datapoint[featureIndex+1:]])
            newNames = np.concatenate([featureNames[:featureIndex], (featureNames[featureIndex+1:])])
        return newdatapoint, newNames

    def get_cutoff_point(data, classes, feature):
        zipped = [(dp[feature], classes[i]) for i, dp in enumerate(data)]
        sort_zipped = sorted(zipped, key=lambda x: (x[0], x[1]))
        prev = sort_zipped[0]
        boundaries = []
        for idx, item in enumerate(sort_zipped):
            if item != prev:
                boundaries.append(idx)
            prev = item
        cutoffGains, cutoffGinis, cutoffPoints = np.zeros(len(boundaries)), np.zeros(len(boundaries)), np.zeros(len(boundaries))
        for cIdx, boundary in enumerate(boundaries):
            testCutoff = sort_zipped[boundary][0]
            cutoffGains[cIdx], cutoffGinis[cIdx] = calc_info_gain(data, classes, feature, testCutoff)
            cutoffPoints[cIdx] = testCutoff
        return cutoffPoints[np.argmin(cutoffGains)]

    def calc_entropy(p):
        return -p*np.log2(p) if p != 0 else 0

    def calc_info_gain(data, classes, feature, cutoffPoint=None):
        gain, ggain = 0, 0
        if cutoffPoint is None:
            values = [*set([datapoint[feature] for datapoint in data])]
        else:
            values = [">=" + str(cutoffPoint), "<" + str(cutoffPoint)]

        featureCounts, entropy, gini = np.zeros(len(values)), np.zeros(len(values)), np.zeros(len(values))

        for valueIndex, value in enumerate(values):
            newClasses = []
            for dataIndex, datapoint in enumerate(data):
                if matches(datapoint[feature], value):
                    featureCounts[valueIndex] += 1
                    newClasses.append(classes[dataIndex])

            classValues, classCounts = np.unique(newClasses, return_counts=True)

            for classIndex in range(len(classValues)):
                entropy[valueIndex] += calc_entropy(
                    float(classCounts[classIndex])/np.sum(classCounts))

                gini[valueIndex] += (float(classCounts[classIndex]
                                           )/np.sum(classCounts))**2

            gain += float(featureCounts[valueIndex]) / \
                nData * entropy[valueIndex]
            ggain += float(featureCounts[valueIndex])/nData * gini[valueIndex]

        return gain, 1-ggain

    maxlevel = kwargs.get("maxlevel", -1)
    forest = kwargs.get("forest", 0)

    data, classes, featureNames = data, targets, features

    nData = len(data)
    level = 0

    dTree = make_tree(data, classes, featureNames, maxlevel, level, forest)

    return lambda x: classify(dTree, x)
