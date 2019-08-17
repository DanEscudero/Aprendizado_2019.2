import scipy.io
import numpy as np
from collections import Counter
from neuralNetwork import neuralNetwork
import random
import operator


def readData():
    # Read raw data
    mat = scipy.io.loadmat('data.mat')
    consts = mat.get('consts')[0][0]

    # Extract useful data
    data = mat.get('mixout').ravel()
    rawLabels = consts[3].ravel()
    labelIndexes = consts[4].ravel()

    # Map label index to character label
    labels = list(map(lambda i: rawLabels[i - 1][0], labelIndexes))

    return (labels, data)


# Returns max length of every points in data
def getMaxLen(data):
    return max(list(map(lambda point: len(point[0]), data)))


def formatData(maxLen, rawData):
    data = []
    for i in range(len(rawData)):
        data.append([])
        for j in range(len(rawData[0])):
            zerosLen = maxLen - len(rawData[i][j])
            data[i].append(np.pad(rawData[i][j], (0, zerosLen), 'constant'))

    # flatten for lists taken from
    # https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-list-of-lists
    def flattenList(l): return [item for sublist in l for item in sublist]

    # Flatten points properly, and make all of them lists instead of ndarrays
    flatData = list(map(flattenList, data))
    return flatData


def main():
    # Get raw data and labels from external file
    (labels, rawData) = readData()

    # Get data filled with 0's, as samples do not have all the same dimension
    data = formatData(getMaxLen(rawData), rawData)

    # Zips data with labels
    labeledData = list(zip(labels, data))

    # This data set is split in 2 parts, both sorted from a-z
    # To assure uniform distribution in both train and test data, data will be shuffled
    random.shuffle(labeledData)

    testRatio = 0.15
    partition = round(testRatio * len(labeledData))

    testingData = labeledData[:partition]
    trainingData = labeledData[partition:]

    # Get counts of labels, to make separation between train and test data easier
    counts = Counter(label for (label, data) in labeledData)

    # Keep in hand all avaliable keys might be useful as well
    avaliableKeys = np.unique(list(counts.elements()))
    avaliableKeys.sort()

    # Define network shape
    inputNodes = len(labeledData[0][1])
    hiddenNodes = 100
    outputNodes = len(avaliableKeys)
    lr = 0.15
    model = neuralNetwork(inputNodes, hiddenNodes, outputNodes, lr)

    firstTest = testingData[0][1]
    q = model.query(firstTest)


if __name__ == "__main__":
    main()
