import scipy.special
import numpy as np


class neuralNetwork:
    def __init__(self, iNodes, hNodes, oNodes, lr):
        # Number of nodes
        self.inputNodes = iNodes
        self.hiddenNodes = hNodes
        self.outputNodes = oNodes

        # Activation function (sigmoid function)
        self.activation = lambda x: scipy.special.expit(x)

        # Learning rate
        self.lr = lr

        self.reset()

        pass

    def reset(self):
        # Initialize weights with random weights in range (-0.5 to 0.5) following normal distribution
        # Weights from input to hidden layer
        self.w_i_h = np.random.normal(
            0.0, pow(self.inputNodes, -0.5), (self.hiddenNodes, self.inputNodes))

        # Weights from hidden to output layer
        self.w_h_o = np.random.normal(
            0.0, pow(self.hiddenNodes, -0.5), (self.outputNodes, self.hiddenNodes))
        pass

    def train(self, inputsList, targetsList):
        inputs = np.array(inputsList, ndmin=2).T
        targets = np.array(targetsList, ndmin=2).T

        # Calculate data past hidden layer
        hiddenInputs = np.dot(self.w_i_h, inputs)
        hiddenOutputs = self.activation(hiddenInputs)

        # Calculate data past output layer
        finalInputs = np.dot(self.w_h_o, hiddenOutputs)
        finalOutputs = self.activation(finalInputs)

        # Calculate errors
        outputError = targets - finalOutputs
        hiddenError = np.dot(self.w_h_o.T, outputError)

        whoAdjustment = np.dot((outputError * finalOutputs * (1-finalOutputs)),
                               np.transpose(hiddenOutputs))
        self.w_h_o += self.lr * whoAdjustment

        wihAdjustment = np.dot((hiddenError * hiddenOutputs * (1-hiddenOutputs)),
                               np.transpose(inputs))
        self.w_i_h += self.lr * wihAdjustment

    def query(self, inputList):
        inputs = np.array(inputList, ndmin=2).T

        # Calculate data past hidden layer
        hiddenInputs = np.dot(self.w_i_h, inputs)
        hiddenOutputs = self.activation(hiddenInputs)

        # Calculate data past output layer
        finalInputs = np.dot(self.w_h_o, hiddenOutputs)
        finalOutputs = self.activation(finalInputs)

        return finalOutputs
