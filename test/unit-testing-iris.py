import sys
from os.path import dirname, abspath, os
sys.path.append(os.path.join(dirname(dirname(__file__)) + '/src'))

import time
import pandas as pd
from sklearn.datasets import load_iris
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
        in dit geval kunnen van de network output kunnen zien of het een setosa of versicolour is aan de hand van de input data
        verder splitten we de data in een train set en een test set
        """

        iris = load_iris()

        df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

        df["target"] = iris.target

        setosa = df["target"].loc[df['target'] == 0]
        versicolour = df["target"].loc[df['target'] == 1]
        setosa_input = df.loc[df['target'] == 0]
        versicolour_input = df.loc[df['target'] == 1]

        big_target_list = setosa.tolist() + versicolour.tolist()
        truth_table = [[x] for x in big_target_list]
        inputs = setosa_input.values.tolist() + versicolour_input.values.tolist()

        inputsTrain, inputsTest, targetsTrain, targetsTest = train_test_split(inputs, truth_table, test_size=0.40, random_state=123)

        truth_output = []

        weights_01 = [random.uniform(-1, 1) for x in range(5)]
        weights_02 = [random.uniform(-1, 1) for x in range(3)]

        hidden_layer_01_neurons = [Neuron(weights_01, random.uniform(-1, 1)) for x in range(3)]
        hidden_layer_01_neurons = [Neuron(weights_02, random.uniform(-1, 1)) for x in range(3)]

        output = Neuron([random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)], random.uniform(-1, 1))

        hidden_layer_01 = NeuronLayer(hidden_layer_01_neurons)
        hidden_layer_02 = NeuronLayer(hidden_layer_01_neurons)
        output_layer = NeuronLayer([output])

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

        print("\nOutput setosa | versicolour | target\n", targetsTest)
        print("\nOutput setosa | versicolour | predicted\n", truth_output)

        print("\n\ntrain score:", neural_network.score(inputsTrain, targetsTrain))
        print("\n\ntest score:", neural_network.score(inputsTest, targetsTest))

        self.assertTrue(truth_table, truth_output)

    def test_versicolour_verginica(self):

        """
        dit functie zorgt ervoor dat de network getest word op de dataset van de iris.
        in dit geval kunnen van de network output kunnen zien of het een versicolour of verginica is aan de hand van de input data
        verder splitten we de data in een train set en een test set
        """

        iris = load_iris()

        df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

        df["target"] = iris.target

        versicolour = df["target"].loc[df['target'] == 1] - 1
        verginica = df["target"].loc[df['target'] == 2] - 1
        versicolour_input = df.loc[df['target'] == 1]
        verginica_input = df.loc[df['target'] == 2]

        big_target_list = versicolour.tolist() + verginica.tolist()
        truth_table = [[x] for x in big_target_list]
        inputs = versicolour_input.values.tolist() + verginica_input.values.tolist()

        inputsTrain, inputsTest, targetsTrain, targetsTest = train_test_split(inputs, truth_table, test_size=0.40, random_state=123)

        truth_output = []

        weights_01 = [random.uniform(-1, 1) for x in range(5)]
        weights_02 = [random.uniform(-1, 1) for x in range(3)]

        hidden_layer_01_neurons = [Neuron(weights_01, random.uniform(-1, 1)) for x in range(3)]
        hidden_layer_01_neurons = [Neuron(weights_02, random.uniform(-1, 1)) for x in range(3)]

        output = Neuron([random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)], random.uniform(-1, 1))

        hidden_layer_01 = NeuronLayer(hidden_layer_01_neurons)
        hidden_layer_02 = NeuronLayer(hidden_layer_01_neurons)
        output_layer = NeuronLayer([output])

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

        print("\nOutput versicolour | verginica | target\n", targetsTest)
        print("\nOutput versicolour | verginica | predicted\n", truth_output)

        print("\n\ntrain score:", neural_network.score(inputsTrain, targetsTrain))
        print("\n\ntest score:", neural_network.score(inputsTest, targetsTest))

        self.assertTrue(truth_table, truth_output)

if __name__ == "__main__":
    unittest.main(verbosity=2)
