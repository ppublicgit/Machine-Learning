import numpy as np
import toolz as tl


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

    def _index(self, key):
        for idx, val in enumerate(self.subtrees):
            if val.key == key:
                return idx
        raise ValueError(f"Key {key} not in subtrees")

    def keys(self):
        return [st.key for st in self.subtrees]

    def subtree(self, key):
        if key not in self.keys():
            return None
        return self.subtrees[self._index(key)].subtrees[0]

def dDTtree(data, targets, features, **kwargs):
    def classify(tree, datapoint):
        nonlocal featureNames
        if tree is None:
            return None
        elif tree.isLeaf():
            return tree
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
        nData = len(data)
        nFeatures = len(data[0])
        newClasses = [*set(classes)]
        frequency = np.zeros(len(newClasses))
        totalEntropy, totalGini = 0, 0

        for i in range(len(newClasses)):
            frequency[i] = count(classes, newClasses)#classes.count(newClasses[i])
            totalEntropy += calc_entropy(float(frequency[i])/nData)
            totalGini += (float(frequency[i])/nData)**2
        totalGini = 1 - totalGini

        if nData == 0 or nFeatures == 0 or (maxlevel >= 0 and level > maxlevel):
            return DecisionTree(newClasses[np.argmax(frequency)], [])
        elif count(classes, classes[0]) == nData:
            return DecisionTree(classes[0], [])
        else:
            # Choose which feature is best
            gain, ggain = np.zeros(nFeatures), np.zeros(nFeatures)
            featureSet = [*range(nFeatures)]
            if forest != 0:
                np.random.shuffle(featureSet)
                featureSet = featureSet[0:forest]
            for featureIndex in featureSet:
                g, gg = calc_info_gain(data, classes, featureIndex)
                gain[featureIndex] = totalEntropy - g
                ggain[featureIndex] = totalGini - gg
            bestFeature = np.argmax(gain)
            tree = DecisionTree(featureNames[bestFeature], [])
            # List the values that bestFeature can take
            values = [*set([datapoint[bestFeature] for datapoint in data])]
            for value in values:
                # Find the datapoints with each feature value
                newData, newClasses = [], []
                for index, datapoint in enumerate(data):
                    if datapoint[bestFeature] == value:
                        newDatapoint, newNames = extract_data(bestFeature, datapoint, featureNames)
                        newData.append(newDatapoint)
                        newClasses.append(classes[index])
                # Now recurse to the next level
                subtree = make_tree(
                    newData, newClasses, newNames, maxlevel, level+1, forest)
                # And on returning, add the subtree on to the tree
                tree.subtrees.append(DecisionTree(value, subtree))
            return tree

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

    def calc_entropy(p):
        return -p*np.log2(p) if p != 0 else 0

    def calc_info_gain(data, classes, feature):
        gain, ggain = 0, 0

        values = [*set([datapoint[feature] for datapoint in data])]

        featureCounts, entropy, gini = np.zeros(len(values)), np.zeros(len(values)), np.zeros(len(values))

        for valueIndex, value in enumerate(values):
            newClasses = []
            for dataIndex, datapoint in enumerate(data):
                if datapoint[feature] == value:
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


def dtree(filename, **kwargs):

    def classify(tree, datapoint):
        nonlocal featureNames
        if type(tree) == type("string"):
            # Have reached a leaf
            return tree
        else:
            a = list(tree.keys())[0]
            for i in range(len(featureNames)):
                if featureNames[i] == a:
                    break
            try:
                t = tree[a][datapoint[i]]
                return classify(t, datapoint)
            except:
                return None

    def classifyAll(tree, data):
        results = []
        for i in range(len(data)):
            results.append(classify(tree, data[i]))
        return results

    def read_data(filename):
        fid = open(filename, "r")
        data = []
        d = []
        for line in fid.readlines():
            d.append(line.strip())
        for d1 in d:
            data.append(d1.split(","))
        fid.close()

        featureNames = data[0]
        featureNames = featureNames[:-1]
        data = data[1:]
        classes = []
        for d in range(len(data)):
            classes.append(data[d][-1])
            data[d] = data[d][:-1]

        return data, classes, featureNames

    def make_tree(data, classes, featureNames, maxlevel=-1, level=0, forest=0):
        """ The main function, which recursively constructs the tree"""
        nData = len(data)
        nFeatures = len(data[0])
        # List the possible classe
        newClasses = []
        for aclass in classes:
            if newClasses.count(aclass) == 0:
                newClasses.append(aclass)
        # Compute the default class (and total entropy)
        frequency = np.zeros(len(newClasses))
        totalEntropy = 0
        totalGini = 0
        index = 0
        for aclass in newClasses:
            frequency[index] = classes.count(aclass)
            totalEntropy += calc_entropy(float(frequency[index])/nData)
            totalGini += (float(frequency[index])/nData)**2
            index += 1
        totalGini = 1 - totalGini
        default = classes[np.argmax(frequency)]
        if nData == 0 or nFeatures == 0 or (maxlevel >= 0 and level > maxlevel):
            # Have reached an empty branch
            return default
        elif classes.count(classes[0]) == nData:
            # Only 1 class remains
            return classes[0]
        else:
            # Choose which feature is best
            gain = np.zeros(nFeatures)
            ggain = np.zeros(nFeatures)
            featureSet = range(nFeatures)
            if forest != 0:
                np.random.shuffle(featureSet)
                featureSet = featureSet[0:forest]
            for feature in featureSet:
                g, gg = calc_info_gain(data, classes, feature)
                gain[feature] = totalEntropy - g
                ggain[feature] = totalGini - gg
            bestFeature = np.argmax(gain)
            tree = {featureNames[bestFeature]: {}}
            # List the values that bestFeature can take
            values = []
            for datapoint in data:
                if datapoint[bestFeature] not in values:
                    values.append(datapoint[bestFeature])
            for value in values:
                # Find the datapoints with each feature value
                newData = []
                newClasses = []
                index = 0
                for datapoint in data:
                    if datapoint[bestFeature] == value:
                        if bestFeature == 0:
                            newdatapoint = datapoint[1:]
                            newNames = featureNames[1:]
                        elif bestFeature == nFeatures:
                            newdatapoint = datapoint[:-1]
                            newNames = featureNames[:-1]
                        else:
                            newdatapoint = datapoint[:bestFeature]
                            newdatapoint.extend(datapoint[bestFeature+1:])
                            newNames = featureNames[:bestFeature]
                            newNames.extend(featureNames[bestFeature+1:])
                        newData.append(newdatapoint)
                        newClasses.append(classes[index])
                    index += 1
                # Now recurse to the next level
                subtree = make_tree(
                    newData, newClasses, newNames, maxlevel, level+1, forest)
                # And on returning, add the subtree on to the tree
                tree[featureNames[bestFeature]][value] = subtree

            return tree

    def calc_entropy(p):
        return -p*np.log2(p) if p != 0 else 0

    def calc_info_gain(data, classes, feature):
        gain = 0
        ggain = 0

        values = []
        for datapoint in data:
            if datapoint[feature] not in values:
                values.append(datapoint[feature])

        featureCounts = np.zeros(len(values))
        entropy = np.zeros(len(values))
        gini = np.zeros(len(values))

        valueIndex = 0

        for value in values:
            dataIndex = 0
            newClasses = []
            for datapoint in data:
                if datapoint[feature] == value:
                    featureCounts[valueIndex] += 1
                    newClasses.append(classes[dataIndex])
                dataIndex += 1

            classValues = []
            for class_ in newClasses:
                if classValues.count(class_) == 0:
                    classValues.append(class_)

            classCounts = np.zeros(len(classValues))
            classIndex = 0
            for classValue in classValues:
                for class_ in newClasses:
                    if class_ == classValue:
                        classCounts[classIndex] += 1
                classIndex += 1

            for classIndex in range(len(classValues)):
                entropy[valueIndex] += calc_entropy(
                    float(classCounts[classIndex])/np.sum(classCounts))
                gini[valueIndex] += (float(classCounts[classIndex]
                                           )/np.sum(classCounts))**2

            gain += float(featureCounts[valueIndex]) / \
                nData * entropy[valueIndex]
            ggain += float(featureCounts[valueIndex])/nData * gini[valueIndex]
            valueIndex += 1

        return gain, 1-ggain

    maxlevel = kwargs.get("maxlevel", -1)
    level = kwargs.get("level", 0)
    forest = kwargs.get("forest", 0)

    data, classes, featureNames = read_data(filename)

    nData = len(data)

    tree = make_tree(data, classes, featureNames, maxlevel, level, forest)

    return lambda x: classify(tree, x)
