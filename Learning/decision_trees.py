import numpy as np
from copy import copy


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
                if val.key == path or (val.key[:4] == "Not:" and val.key[4:] != path):
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

    def _check_Not(self):
        for st in self.subtrees:
            if isinstance(st.key, str) and st.key[:4] == "Not:":
                return True
        return False

    def subtree(self, path):
        if isinstance(path, str):
            if path not in self.keys() and not self._check_Not():
                return None
            return self.subtrees[self._index(path, True)].subtrees[0]
        else:
            return self.subtrees[self._index(path, False)].subtrees[0]


def dtree(data, targets, features, **kwargs):

    def classify(tree, datapoint):
        if datapoint is None:
            return tree
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
        if outtype == "classification":
            newClasses = [*set(classes)]
        totalGain, frequency = calc_total_gain(classes)

        if outtype == "classification" and (nData == 0 or nFeatures == 0 or (maxlevel >= 0 and level > maxlevel)):
            return DecisionTree(newClasses[np.argmax(frequency)], [])
        elif outtype == "classification" and count(classes, classes[0]) == nData:
            return DecisionTree(classes[0], [])
        elif outtype == "regression" and nData < minData:
            return DecisionTree(np.mean(classes), [])
        else:
            # Choose which feature is best
            gain, cutoffPoints = np.zeros(nFeatures), [None]*nFeatures
            featureSet = [*range(nFeatures)]
            if forest != 0:
                np.random.shuffle(featureSet)
                featureSet = featureSet[0:forest]
            for featureIndex in featureSet:
                if treeType == "ID3":
                    cutoffPoint = None
                elif isinstance(data[0][featureIndex], str):
                    cutoffPoint = get_cutoff_choice(data, classes, featureIndex)
                else:
                    cutoffPoint = get_cutoff_point(data, classes, featureIndex)
                gain[featureIndex] = totalGain - calc_info_gain(data, classes, featureIndex, cutoffPoint)
                cutoffPoints[featureIndex] = cutoffPoint
            bestFeature = np.argmax(gain)
            binary = True
            if isinstance(cutoffPoints[bestFeature], str):
                values = [*set([datapoint[bestFeature] for datapoint in data])]
                if len(values) > 2:
                    values = [cutoffPoints[bestFeature], "Not:"+cutoffPoints[bestFeature]]
                    binary = False
            else:
                values = [">=" + str(cutoffPoints[bestFeature]), "<" + str(cutoffPoints[bestFeature])]
            tree = DecisionTree(featureNames[bestFeature], [])
            # List the values that bestFeature can take
            for value in values:
                # Find the datapoints with each feature value
                newData, newClasses, newNames  = [], [], []
                for index, datapoint in enumerate(data):
                    if (binary or value[:4] != "Not:") and matches(datapoint[bestFeature], value):
                        newDatapoint, newNames = extract_data(bestFeature, datapoint, featureNames)
                        newData.append(newDatapoint)
                        newClasses.append(classes[index])
                    elif matches(datapoint[bestFeature], value):
                        newData.append(datapoint)
                        newClasses.append(classes[index])
                        newNames = copy(featureNames)
                # Now recurse to the next level
                if outtype == "regression" and check_coeff_variation(newClasses):
                    return DecisionTree(np.mean(newClasses), [])
                subtree = make_tree(
                    newData, newClasses, newNames, maxlevel, level+1, forest)
                # And on returning, add the subtree on to the tree
                tree.subtrees.append(DecisionTree(value, subtree))
            return tree

    def check_coeff_variation(classes):
        std = 0
        nClasses = len(classes)
        meanClasses = np.mean(classes)
        for i in range(nClasses):
            std += np.sqrt((classes[i] - meanClasses)**2)
        std /= np.sqrt(nClasses)
        coeffVar = std/meanClasses*100.0
        return coeffVar < minCoeffVar

    def matches(datapointVal, checkVal):
        if isinstance(datapointVal, str):
            if checkVal[:4] == "Not:":
                return not datapointVal == checkVal[4:]
            else:
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
        if outtype == "classification":
            zipped = [(dp[feature], classes[i]) for i, dp in enumerate(data)]
            sort_zipped = sorted(zipped, key=lambda x: (x[0], x[1]))
            prev = sort_zipped[0]
            boundaries = []
            for idx, item in enumerate(sort_zipped):
                if item != prev:
                    boundaries.append(idx)
                prev = item
        else:
            zipped = [dp[feature] for dp in data]
            sort_zipped = []
            minLen = 8
            if feature == 3:
                pass
                #breakpoint()
            if len(zipped) > minLen:
                [sort_zipped.append((np.quantile(zipped, float(i)/minLen), None)) for i in range(1, minLen)]
            else:
                sort_zipped = [(zipped[i], None) for i in range(len(zipped))]
            sort_zipped = [*set(sort_zipped)]
            boundaries = [*range(len(sort_zipped))]
        cutoffGain, cutoffPoints = np.zeros(len(boundaries)), np.zeros(len(boundaries))
        for cIdx, boundary in enumerate(boundaries):
            testCutoff = sort_zipped[boundary][0]
            cutoffGain[cIdx] = calc_info_gain(data, classes, feature, testCutoff)
            cutoffPoints[cIdx] = testCutoff
        return cutoffPoints[np.argmin(cutoffGain)]

    def get_cutoff_choice(data, classes, feature):
        choices = [dp[feature] for dp in data]
        choices = [*set(choices)]
        if len(choices) <= 2:
            return ""
        cutoffGain, options = np.zeros(len(choices)), np.zeros(len(choices))
        for cIdx, choice in enumerate(choices):
            cutoffGain[cIdx] = calc_info_gain(data, classes, feature, choice)
        return choices[np.argmin(cutoffGain)]

    def calc_total_gain(classes):
        totalGain = 0
        if treeType == "ID3" and outtype == "classification":
            newClasses = [*set(classes)]
            frequency = np.zeros(len(newClasses))
            for i in range(len(newClasses)):
                frequency[i] = count(classes, newClasses[i])
                totalGain += calc_entropy(float(frequency[i])/nData)
        elif treeType == "CART" and outtype == "classification":
            newClasses = [*set(classes)]
            frequency = np.zeros(len(newClasses))
            for i in range(len(newClasses)):
                frequency[i] = count(classes, newClasses[i])
                totalGain += (float(frequency[i])/nData)**2
            totalGain = 1 - totalGain
        elif treeType == "CART" and outtype == "regression":
            totalStd = 0
            meanClass = np.mean(classes)
            frequency = None
            for i in range(len(classes)):
                totalStd += np.sqrt((classes[i]-meanClass)**2)
            totalStd /= np.sqrt(nData)
            totalGain = 1 - totalStd
        else:
            raise ValueError("Invalid treeType outtype combination")
        return totalGain, frequency

    def calc_info_gain(data, classes, feature, cutoff=None):
        if outtype == "classification" and treeType == "CART":
            return calc_info_gini(data, classes, feature, cutoff)
        elif outtype == "classification" and treeType == "ID3":
            return calc_info_entropy(data, classes, feature, cutoff)
        elif outtype == "regression" and treeType == "CART":
            return calc_info_sse(data, classes, feature, cutoff)
        else:
            raise ValueError(f"Invalid treeType {treeType} with outtype {outtype}")

    def calc_info_sse(data, classes, feature, cutoff):
        gain = 0
        nData = len(data)
        if isinstance(cutoff, str):
            values = [*set([datapoint[feature] for datapoint in data])]
            if len(values) > 2:
                values = [cutoff, "Not:"+cutoff]
        else:
            values = [">=" + str(cutoff), "<" + str(cutoff)]

        featureCounts, sse = np.zeros(len(values)), np.zeros(len(values))

        for valueIndex, value in enumerate(values):
            newClasses = []
            for dataIndex, datapoint in enumerate(data):
                if matches(datapoint[feature], value):
                    featureCounts[valueIndex] += 1
                    newClasses.append(classes[dataIndex])
            nSplit = float(len(newClasses))
            pSplit = nSplit/float(nData)
            stdSplit = 0
            meanSplit = np.mean(newClasses)
            for i in range(len(newClasses)):
                stdSplit += np.sqrt((newClasses[i] - meanSplit)**2)
            stdSplit /= np.sqrt(nSplit)
            gain += stdSplit * pSplit
        return gain

    def calc_info_gini(data, classes, feature, cutoff=None):
        gain = 0
        if isinstance(cutoff, str):
            values = [*set([datapoint[feature] for datapoint in data])]
            if len(values) > 2:
                values = [cutoff, "Not:"+cutoff]
        else:
            values = [">=" + str(cutoff), "<" + str(cutoff)]

        featureCounts, gini = np.zeros(len(values)), np.zeros(len(values))

        for valueIndex, value in enumerate(values):
            newClasses = []
            for dataIndex, datapoint in enumerate(data):
                if matches(datapoint[feature], value):
                    featureCounts[valueIndex] += 1
                    newClasses.append(classes[dataIndex])

            classValues, classCounts = np.unique(newClasses, return_counts=True)

            for classIndex in range(len(classValues)):
                gini[valueIndex] += (float(classCounts[classIndex]
                                           )/np.sum(classCounts))**2

            gain += float(featureCounts[valueIndex])/nData * gini[valueIndex]
        return 1-gain

    def calc_entropy(p):
        return -p*np.log2(p) if p != 0 else 0

    def calc_info_entropy(data, classes, feature, cutoff=None):
        gain = 0
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

            gain += float(featureCounts[valueIndex]) / \
                nData * entropy[valueIndex]
        return gain

    maxlevel = kwargs.get("maxlevel", -1)
    forest = kwargs.get("forest", 0)
    outtype = kwargs.get("outtype", "classification")
    minCoeffVar = kwargs.get("coeff_var", 5.)
    treeType = kwargs.get("treeType", "CART")
    minData = kwargs.get("min_cutoff", -1)

    if outtype == "regression" and not treeType == "CART":
        raise ValueError("Only treeType CART supports regression")
    elif outtype not in ["regression", "classification"]:
        raise ValueError("Invalid outtype")
    elif treeType not in ["CART", "ID3"]:
        raise ValueError("Invalid treeType")

    data, classes, featureNames = data, targets, features

    nData = len(data)
    level = 0

    dTree = make_tree(data, classes, featureNames, maxlevel, level, forest)

    return lambda x: classify(dTree, x)
