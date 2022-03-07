class NeuronLayer:
    def __init__(self, neurons):
        self.neurons = neurons
        self.errors = []
        self.weights = [neuron.get_weights() for neuron in self.neurons]

    def get_neurons(self):

        """
        functie om de neurons te krijgen binnen een layer

        """

        return [neuron for neuron in self.neurons]

    def error(self, output, next_layer_weights, next_layer_error):
        self.errorNextNeurons = []  # Reset the errors of the neurons of the next layer
        if next_layer_weights and next_layer_error:
            for i in range(len(self.neurons)):
                nextWeights = [weights[i] for weights in next_layer_weights]
                self.neurons[i].neuron_error(output, nextWeights, next_layer_error)
                self.errors.append(self.neurons[i].error)
        else:
            for i in range(len(self.neurons)):
                self.neurons[i].neuron_error(output[i], [], [])
                self.errors.append(self.neurons[i].error)


    def learn(self, eta):
        for neuron in self.neurons:
            neuron.learn(eta)
        self.errors = []
        self.weights = [neuron.weights for neuron in self.neurons]

    def update(self):
        for neuron in self.neurons:
            neuron.update()

    def get_errors(self):
        return self.errors

    def get_weights(self):
        return self.weights

    def __str__(self):

        """
        dit functie gebruiken we om informatie te krijgen van de layer object
        """

        return f"neuronen: {self.neurons}, errors: {self.errors}"
