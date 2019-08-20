import sys
import math
import random
import scipy.io
import operator
import numpy as np
import matplotlib.pyplot as plt
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


# Returns the amount of unique labels in an array of tuples (label, data)
# Has high complexity and could be slow for too large arrays
def amountOfLabels(data):
    counts = Counter(label for (label, data) in data)
    return len(np.unique(list(counts.elements())))


def partitionData(ratio, allData):
    # Get total amount of labels, in order to check if training data has all the labels in it
    totalLabels = amountOfLabels(allData)

    partitionIndex = round(ratio * len(allData))

    testingData = allData[:partitionIndex]
    trainingData = allData[partitionIndex:]

    # Array is shuffled again until both testing and training data has all the labels
    reshuffle = (amountOfLabels(testingData) != totalLabels) or (
        amountOfLabels(trainingData) != totalLabels)
    while (reshuffle):
        random.shuffle(allData)
        testingData = allData[:partitionIndex]
        trainingData = allData[partitionIndex:]

        reshuffle = (amountOfLabels(testingData) != totalLabels) or (
            amountOfLabels(trainingData) != totalLabels)

    return (testingData, trainingData, allData)


def getSplitData(ratio):
     # Get raw data and labels from external file
    (labels, rawData) = readData()

    # Get data filled with 0's, as samples do not have all the same dimension
    data = formatData(getMaxLen(rawData), rawData)

    # Zips data with labels
    allData = list(zip(labels, data))

    # This data set is split in 2 parts, both sorted from a-z
    # To assure uniform distribution in both train and test data, data will be shuffled
    random.shuffle(allData)

    # Split data into training and testing
    return partitionData(ratio, allData)


def setupModel(allData):
    lr = 0.15
    inputNodes = len(allData[0][1])
    hiddenNodes = 100
    outputNodes = amountOfLabels(allData)
    return neuralNetwork(inputNodes, hiddenNodes, outputNodes, lr)


def main():
    # As there are 2.8k examples, 15% of them results in ~400 examples for testing
    testRatio = 0.15
    (testingData, trainingData, allData) = getSplitData(testRatio)

    # Get counts of labels, to make separation between train and test data easier
    counts = Counter(label for (label, data) in allData)

    # Keep in hand all avaliable keys sorted lexicographically
    avaliableKeys = np.unique(list(counts.elements()))
    avaliableKeys.sort()

    # Associate keys to indexes in a dictionary in the form: { a: 0, b: 1, ... }
    indexedKeys = []
    for i in range(len(avaliableKeys)):
        char = avaliableKeys[i]
        indexedKeys.append((char, i))

    dictLabels = dict(indexedKeys)

    # Create model
    model = setupModel(allData)

    epochs = 1

    # Train model
    trainModel(model, epochs, trainingData, dictLabels)

    # Test model
    (correctGuesses, accuracy) = testModel(model, testingData, avaliableKeys)

    print('Model got', correctGuesses, 'out of', len(testingData))
    print('Accuracy:', 100 * round(accuracy, 2), '%')


def chunks(l, nChunks):
    chunkSize = math.ceil(len(l) / nChunks)
    return list(l[i:i+chunkSize] for i in range(0, len(l), chunkSize))


def crossValidateEpochs():
    nChunks = 10
    testRatio = 0.15
    (testingData, trainingData, allData) = getSplitData(testRatio)
    trainDataChunks = chunks(testingData, nChunks)

    counts = Counter(label for (label, data) in allData)

    avaliableKeys = np.unique(list(counts.elements()))
    avaliableKeys.sort()

    indexedKeys = []
    for i in range(len(avaliableKeys)):
        char = avaliableKeys[i]
        indexedKeys.append((char, i))

    dictLabels = dict(indexedKeys)

    model = setupModel(allData)

    epochsOptions = list(map(lambda x: (x + 1), range(nChunks)))
    print(epochsOptions)

    bestEpochs = 0
    bestAccuracy = 0
    accuracies = []

    for i in range(len(epochsOptions)):
        epochs = epochsOptions[i]
        currentTestingData = []
        currentTrainingData = []
        for j in range(nChunks):
            if (i != j):
                currentTrainingData.extend(trainDataChunks[j - 1])
            else:
                currentTestingData.extend(trainDataChunks[j - 1])

        # Reset model, which might be trained from last round
        model.reset()

        # train model with current number of epochs
        trainModel(model, epochs, currentTrainingData, dictLabels)

        # test model with the split of data that wasn't used for training
        (_, accuracy) = testModel(model, currentTestingData, avaliableKeys)

        print(epochs, accuracy)

        accuracies.append(accuracy)

        if (accuracy > bestAccuracy):
            bestEpochs = epochs

    # Constrain graph axis
    plt.xlim(0, max(epochsOptions))
    plt.ylim(0, 1.1)

    plt.title('Iterações sobres os dados vs. Acurácia do modelo')
    plt.xlabel('Quantidade de iterações pelos dados de treino')
    plt.ylabel('Acurácia do modelo')

    # Plot epochs and accuracies
    plt.plot(epochsOptions, accuracies)

    plt.show()


if __name__ == "__main__":
    if (sys.argv[1] == 'crossValidate'):
        crossValidateEpochs()
    else:
        main()
