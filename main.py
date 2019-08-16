import scipy.io
import numpy as np
from collections import Counter
from neuralNetwork import neuralNetwork


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
def getMaxLen(labeledData):
    return max(list(map(lambda point: len(point[0]), labeledData)))


def main():
    (labels, rawData) = readData()

    maxPointLen = getMaxLen(rawData)

    data = []
    for i in range(len(rawData)):
        for j in range(len(rawData[0])):
            data.append([])
            zerosLen = maxPointLen - len(rawData[i][j])
            data[i].append(np.pad(rawData[i][j], (0, zerosLen), 'constant'))

    labeledData = zip(labels, data)


if __name__ == "__main__":
    main()
