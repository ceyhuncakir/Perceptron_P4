import sys
from os.path import dirname, abspath, os
sys.path.append(os.path.join(dirname(dirname(__file__)) + '/src'))

import time
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import normalize
import random
import unittest
from Neuron import Neuron
from NeuronLayer import NeuronLayer
from NeuronNetwork import NeuronNetwork
from sklearn.model_selection import train_test_split


class IrisTesting(unittest.TestCase):
    def test_setosa_versicolour(self):

        """
        dit functie zorgt ervoor dat de network getest word op de dataset van de iris.
        in dit geval kunnen van de network output kunnen zien of het een setosa, versicolour, of virginica is aan de hand van de input data
        verder splitten we de data in een train set en een test set
        """

        iris = load_iris()

        X = iris.data
        Y = pd.get_dummies(iris.target)

        big_input_list = normalize(X.tolist())
        big_target_list = Y.values.tolist()

        inputsTrain, inputsTest, targetsTrain, targetsTest = train_test_split(big_input_list, big_target_list, test_size=0.40, random_state=123)

        truth_output = []

        weights_01 = [random.uniform(-1, 1) for x in range(5)]
        weights_02 = [random.uniform(-1, 1) for x in range(3)]

        hidden_layer_01_neurons = [Neuron(weights_01, random.uniform(-1, 1)) for x in range(3)]
        hidden_layer_01_neurons = [Neuron(weights_02, random.uniform(-1, 1)) for x in range(3)]

        hidden_layer_output_neurons = [Neuron(weights_02, random.uniform(-1, 1)) for x in range(3)]

        hidden_layer_01 = NeuronLayer(hidden_layer_01_neurons)
        hidden_layer_02 = NeuronLayer(hidden_layer_01_neurons)
        output_layer = NeuronLayer(hidden_layer_output_neurons)

        neural_network = NeuronNetwork([hidden_layer_01, output_layer])

        start = time.time()
        for epoch in range(10000):
            neural_network.train(inputsTrain, targetsTrain, 0.1)
        print("\n\ntraining done, time taken:", (time.time() - start))

        for input in range(len(inputsTest)):
            output = neural_network.feed_forward(inputsTest[input])

            if output[0] < 0.5:
                truth_output.append([0])
            else:
                truth_output.append([1])

        print("\nOutput setosa | versicolour | virginica | target\n", targetsTest)
        print("\nOutput setosa | versicolour | virginica | predicted\n", truth_output)

        print("\n\ntrain score:", neural_network.score(inputsTrain, targetsTrain))
        print("\n\ntest score:", neural_network.score(inputsTest, targetsTest))


if __name__ == "__main__":
    unittest.main(verbosity=2)
