class NeuronLayer:
    def __init__(self, neurons):
        self.neurons = neurons
        self.errors = []
        self.weights = [neuron.get_weights() for neuron in self.neurons]

    def get_neurons(self):

        """
        functie om de neurons te krijgen binnen een layer

        returns: (list) neurons
        """

        return [neuron for neuron in self.neurons]

    def error(self, output, next_layer_weights, next_layer_error):

        """
        dit functie berekent de error voor de layer met neuronen.

        params:
            (int) output
            (list) next_layer_weights
            (list) next_layer_error
        """

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

        """
        dit functie zorgt er voor dat voor elke neuron binnen de layer de learning doet.
        de uiteindelijke weights worden gestopt in de lijst van weights

        params: (float) eta
        """

        for neuron in self.neurons:
            neuron.learn(eta)
        self.errors = []
        self.weights = [neuron.weights for neuron in self.neurons]

    def update(self):

        """
        dit functie zorgt er voor dat de bias en de weight van een neuron geupdated word.
        """

        for neuron in self.neurons:
            neuron.update()

    def get_errors(self):

        """
        dit functie zorgt er voor dat de errors terug gereturned word

        returns: (list) errors
        """

        return self.errors

    def get_weights(self):


        """
        dit functie zorgt er voor dat de weights terug gereturned worden

        returns: (list) weights
        """

        return self.weights

    def __str__(self):

        """
        dit functie gebruiken we om informatie te krijgen van de layer object
        """

        return f"neuronen: {self.neurons} / errors: {self.errors}"
