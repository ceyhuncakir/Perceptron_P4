import sys
from os.path import dirname, abspath, os
sys.path.append(os.path.join(dirname(dirname(__file__)) + '/src'))

import time
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import unittest
from Neuron import Neuron
from NeuronLayer import NeuronLayer
from NeuronNetwork import NeuronNetwork


class DigitTesting(unittest.TestCase):
    def test_digits(self):

        """
        dit functie zorgt ervoor dat de network getest word op de dataset van de digit set.
        aangezien de target groter dan een 1 kan zijn gebruiken we hiervoor de pd.get_dummies functie ervoor om een 0 of een 1 te krijgen.
        verder splitten we de data in een train set en een test set
        """

        data = load_digits(as_frame=True)

        X = data.data
        Y = pd.get_dummies(data.target)

        big_input_list = normalize(X.values.tolist())
        big_target_list = Y.values.tolist()

        output_true = []

        inputsTrain, inputsTest, targetsTrain, targetsTest = train_test_split(big_input_list, big_target_list, test_size=0.20, random_state=123)
        weights = [random.uniform(-1, 1) for x in range(64)]
        weights_01 = [random.uniform(-1, 1) for x in range(20)]
        weights_02 = [random.uniform(-1, 1) for x in range(20)]

        hidden_layer_01_neurons = [Neuron(weights, random.uniform(-1, 1)) for x in range(20)]
        hidden_layer_02_neurons = [Neuron(weights_01, random.uniform(-1, 1)) for x in range(20)]
        output_layer_neurons = [Neuron(weights_02, random.uniform(-1, 1)) for x in range(10)]


        hidden_layer_01 = NeuronLayer(hidden_layer_01_neurons)
        hidden_layer_02 = NeuronLayer(hidden_layer_02_neurons)
        output_layer = NeuronLayer(output_layer_neurons)

        neural_network = NeuronNetwork([hidden_layer_01, hidden_layer_02, output_layer])

        start = time.time()
        for epoch in range(200):
            neural_network.train(inputsTrain, targetsTrain, 0.1)
        print("\n\ntraining done, time taken:", (time.time() - start))

        for input in range(len(inputsTest)):
            output = neural_network.feed_forward(inputsTest[input])

            case = []
            for singleoutput in output:
                if singleoutput < 0.5:
                    singleoutput = 0
                    case.append(singleoutput)
                else:
                    singleoutput = 1
                    case.append(singleoutput)

                output_true.append(case)

        print("\n\ntrain score:", neural_network.score(inputsTrain, targetsTrain))
        print("\n\ntest score:", neural_network.score(inputsTest, targetsTest))


if __name__ == "__main__":
    unittest.main(verbosity=2)
