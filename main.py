import scipy.io
import numpy
from collections import Counter

from neuralNetwork import neuralNetwork

mat = scipy.io.loadmat('data.mat')

# print all keys
# print(mat.keys())

mixout = mat.get('mixout')[0]  # len 2858

consts = mat.get('consts')[0][0]
# print(consts.shape)
# print(consts.size)
# print(consts)

indexes = [123, 122, 543, 541, 256, 909, 111]

keys = consts[3].ravel()
labels = consts[4].ravel()

# for mixoutChar in mixout:
#     fst = len(mixoutChar[0]) == len(mixoutChar[1])
#     snd = len(mixoutChar[1]) == len(mixoutChar[2])
#     cond = fst and snd
#     if (not cond):
#         print('aaaaaaaaaaaaaaaaa')

# print(len(consts[4][0]))
# for x in consts:
#     print(x)


def main():
    input_nodes = 3
    hidden_nodes = 3
    output_nodes = 3

    learning_rate = 0.3

    n = neuralNetwork(3, 3, 3, 0.3)


if __name__ == "__main__":
    main()
