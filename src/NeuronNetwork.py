class NeuronNetwork:
    def __init__(self, layers):
        self.neuron_layers = layers
        self.inputs = None

    def feed_forward(self, inputs):

        """
        functie die van de gegeven input door het netwerk laat lopen om tot een uitkomst te komen

        params: (list) inputs
        returns: (list) output
        """
        self.inputs = inputs
        inputs = self.inputs
        act_output = None

        for layer in self.neuron_layers:
            new_output = []
            for neuron in layer.get_neurons():
                new_output.append(neuron.calculate(inputs))

            inputs = new_output
            act_output = new_output

        return act_output

    def train(self, inputs, targets, eta):

        """
        dit functie zorgt er voor dat de nn backpropagated word

        params:
            (list) inputs,
            (list) targets,
            (float) eta
        """

        for input in range(len(inputs)):
            self.feed_forward(inputs[input])
            for layer in range(len(self.neuron_layers)):
                layer = (layer * - 1)
                if layer == 0:
                    self.neuron_layers[layer - 1].error(targets[input], [], [])
                else:
                    self.neuron_layers[layer - 1].error(targets[input], self.neuron_layers[layer].weights, self.neuron_layers[layer].errors)
            self.update_network(eta)

    def update_network(self, eta):

        """
        dit functie zorgt er voor dat de layers leeren en updaten met de bepaalde errors
        params: (float) eta
        """

        for layer in range(len(self.neuron_layers)):
            layer = (layer * - 1)
            self.neuron_layers[layer - 1].learn(eta)
            self.neuron_layers[layer - 1].update()

    def score(self, inputs, targets):

        """
        dit functie calculeert de score tussen de inputs en targets
        params:
            (list) inputs,
            (list) targets
        """

        equal = 0
        for i in range(len(inputs)):
            output = self.feed_forward(inputs[i])
            if output.index(max(output)) == targets[i].index(max(targets[i])):
                equal += 1
        return equal / len(targets) * 100


    def __str__(self):

        """
        functie om informatie van de metwork te krijgen
        """

        return f"layers: {self.neuron_layers} / inputs: {self.inputs}"
