import math

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.new_weights = []
        self.bias = bias
        self.new_bias = None
        self.inputs = []
        self.error = None
        self.output = None

    def sigmoid(self, x):

        """
        de sigmoid functie om de weighting te berekenen

        params: (int) x
        """

        return 1 / (1 + math.exp(-x))

    def calculate(self, inputs):

        """
        de calculate functie om van een input een output te maken

        params: (list) inputs
        """
        self.inputs = inputs
        output = self.sigmoid(sum(x*y for x, y in zip(inputs, self.weights)) + self.bias)
        self.output = output
        return output

    def neuron_error(self, output, next_neuron_weights, next_neuron_error):

        """
        dit functie zet de error van de neuron

        params:
            (int) output
            (list) next_neuron_weights
            (list) next_neuron_error
        """

        if next_neuron_weights and next_neuron_error:
            sumFromNextNodes = 0
            for i in range(len(next_neuron_weights)):
                sumFromNextNodes += next_neuron_weights[i] * next_neuron_error[i]

            self.error = self.output * (1 - self.output) * sumFromNextNodes
        else:
            self.error = self.output * (1 - self.output) * -(output - self.output)


    def learn(self, eta):
        """
        dit functie leert de neuron

        params: (float) eta
        """

        self.new_weights = [self.weights[i] - eta * self.inputs[i] * self.error for i in range(len(self.weights))]
        self.new_bias = self.bias - eta * self.error

    def update(self):

        """
        dit functie update de bias en de weights met de nieuwe weights

        """

        if self.new_weights:
            self.weights = self.new_weights
            self.bias = self.new_bias
        self.new_weights = []
        self.new_bias = None

    def get_weights(self):

        """
        functie om de weights te krijgen van de neuron

        returns: (list) weights
        """

        return self.weights

    def get_bias(self):

        """
        functie om de bias te krijgen van de neuron

        returns: (list) bias
        """

        return self.bias

    def get_error(self):

        """

        dit functie returned de error van de neuron

        returns: (float) error
        """

        return self.error


    def get_output(self):

        """
        dit functie returned de output

        returns: (float) output
        """

        return self.output

    def __str__(self):

        """
        functie om informatie van de neuron te krijgen.
        """

        return f"input: {self.inputs} / weights: {self.weights} / bias: {self.bias} / output: {self.output} / error: {self.error}"
