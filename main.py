import sys
import random
import scipy.io
import operator
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


def trainModel(model, epochs, trainingData, dictLabels):
    for epoch in range(epochs):
        for (label, data) in trainingData:
            # Create targets: all values are 0.01, except the correct one, which is 1
            targets = np.zeros(len(dictLabels)) + 0.01
            targets[dictLabels.get(label)] = 1
            model.train(data, targets)
            pass
        pass


def testModel(model, testingData, labels):
    correctGuesses = 0
    for (label, data) in testingData:
        # Query returns an array of probabilities, where 0th is 'a', 1st is 'b'...
        guesses = model.query(data)

        index = np.argmax(guesses)
        guess = labels[index]

        # If label matches correct one, sum to correctGuesses
        if (guess == label):
            correctGuesses = correctGuesses + 1
    pass

    accuracy = correctGuesses / len(testingData)
    return (correctGuesses, accuracy)


def getSplitData(ratio):
     # Get raw data and labels from external file
    (labels, rawData) = readData()

    # Get data filled with 0's, as samples do not have all the same dimension
    data = formatData(getMaxLen(rawData), rawData)

    # Zips data with labels
    labeledData = list(zip(labels, data))

    # This data set is split in 2 parts, both sorted from a-z
    # To assure uniform distribution in both train and test data, data will be shuffled
    random.shuffle(labeledData)

    # Split data into training and testing
    partition = round(ratio * len(labeledData))

    testingData = labeledData[:partition]
    trainingData = labeledData[partition:]
    return (testingData, trainingData, labeledData)


def main():
    # As there are 2.8k examples, 15% of them results in ~400 examples for testing
    testRatio = 0.15
    (testingData, trainingData, allData) = getSplitData(testRatio)

    # Get counts of labels, to make separation between train and test data easier
    counts = Counter(label for (label, data) in allData)

    # Keep in hand all avaliable keys sorted lexicographically
    avaliableKeys = np.unique(list(counts.elements()))
    avaliableKeys.sort()

    # Associate keys to indexes
    indexedKeys = []
    for i in range(len(avaliableKeys)):
        char = avaliableKeys[i]
        indexedKeys.append((char, i))

    dictLabels = dict(indexedKeys)

    # Define network shape and hyperparameters
    lr = 0.15
    inputNodes = len(allData[0][1])
    hiddenNodes = 100
    outputNodes = len(avaliableKeys)
    model = neuralNetwork(inputNodes, hiddenNodes, outputNodes, lr)

    epochs = 1

    # Train model
    trainModel(model, epochs, trainingData, dictLabels)

    # Test model
    (correctGuesses, accuracy) = testModel(model, testingData, avaliableKeys)

    print('Model got', correctGuesses, 'out of', len(testingData))
    print('Accuracy:', 100 * round(accuracy, 2), '%')


def chunks(l, nChunks):
    nChunks = max(1, nChunks)
    return list(l[i:i+nChunks] for i in range(0, len(l), nChunks))


def crossValidateEpochs():
    nChunks = 10
    testRatio = 0.15
    (testingData, trainingData, labeledData) = getSplitData(testRatio)
    trainDataChunks = chunks(testingData, nChunks)

    epochsOptions = list(range(1, nChunks + 1))

    # TODO cross validation for number of epochs
    for epochs in epochsOptions:
        for i in range(nChunks):
            # train is all chunks except the one with index i
            # test is the chunk whose index is i
            # resetModel
            # trainModel
            # testModel
            # plot (epochs, accuracy)
            # if accuracy > bestAccuracy: best epochs = epochs
            pass
        pass


if __name__ == "__main__":
    # crossValidateEpochs()
    main()
