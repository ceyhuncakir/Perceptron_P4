import sys
from os.path import dirname, abspath, os
sys.path.append(os.path.join(dirname(dirname(__file__)) + '/src'))

import random
import unittest
from Neuron import Neuron
from NeuronLayer import NeuronLayer
from NeuronNetwork import NeuronNetwork


class NeuronNetworkTesting(unittest.TestCase):

    def test_and_gate(self):

        """
        dit is de test voor de and gate. aangezien de output als een float komt tussen 0 en 1.
        doen we een check of het groter dan 0.5 is waardoor we een 0 of een 1 kunnen geven
        """

        combination = [[0, 0], [1, 0], [0, 1], [1, 1]]
        truth_table = [[0], [0], [0], [1]]
        truth_output = []

        n1 = Neuron([random.uniform(-1, 1), random.uniform(-1, 1)], random.uniform(-1, 1))

        hidden_layer_01 = NeuronLayer([n1])
        neural_network = NeuronNetwork([hidden_layer_01])

        for epoch in range(10000):
            neural_network.train(combination, truth_table, 0.1)

        for input in range(len(combination)):
            output = neural_network.feed_forward(combination[input])
            truth_output.append(output)

        self.assertTrue(truth_table, truth_output)

    def test_xor_gate(self):


        """
        dit is de test voor de xor gate aangezien de output als een float komt tussen 0 en 1.
        doen we een check of het groter dan 0.5 is waardoor we een 0 of een 1 kunnen geven
        """

        combination = [[0, 0], [0, 1], [1, 0], [1, 1]]
        truth_table = [[0], [1], [1], [0]]
        truth_output = []

        n1 = Neuron([random.uniform(-1, 1), random.uniform(-1, 1)], random.uniform(-1, 1))
        n2 = Neuron([random.uniform(-1, 1), random.uniform(-1, 1)], random.uniform(-1, 1))
        output = Neuron([random.uniform(-1, 1), random.uniform(-1, 1)], random.uniform(-1, 1))

        hidden_layer_01 = NeuronLayer([n1, n2])
        output_layer = NeuronLayer([output])
        neural_network = NeuronNetwork([hidden_layer_01, output_layer])

        for epoch in range(10000):
            neural_network.train(combination, truth_table, 0.1)

        for input in range(len(combination)):
            output = neural_network.feed_forward(combination[input])
            truth_output.append(output)

        self.assertTrue(truth_table, truth_output)

    def test_half_adder_gate(self):

        """
        dit is de test voor de half adder gate aangezien de output als een float komt tussen 0 en 1.
        doen we een check of het groter dan 0.5 is waardoor we een 0 of een 1 kunnen geven
        """

        combination = [[0, 0], [0, 1], [1, 0], [1, 1]]
        truth_table = [[0, 0], [1, 0], [1, 0], [0, 1]]
        truth_output = []

        n1 = Neuron([random.uniform(-1, 1), random.uniform(-1, 1)], random.uniform(-1, 1))
        n2 = Neuron([random.uniform(-1, 1), random.uniform(-1, 1)], random.uniform(-1, 1))
        n3 = Neuron([random.uniform(-1, 1), random.uniform(-1, 1)], random.uniform(-1, 1))

        summ = Neuron([random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)], random.uniform(-1, 1))
        carry = Neuron([random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)], random.uniform(-1, 1))

        hidden_layer_01 = NeuronLayer([n1, n2, n3])
        output_layer = NeuronLayer([summ, carry])
        neural_network = NeuronNetwork([hidden_layer_01, output_layer])

        for epoch in range(10000):
            neural_network.train(combination, truth_table, 0.1)

        for input in range(len(combination)):
            output = neural_network.feed_forward(combination[input])
            truth_output.append(output)

        self.assertTrue(truth_table, truth_output)

if __name__ == "__main__":
    unittest.main(verbosity=2)
