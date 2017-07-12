class NeuralNetwork(object):

    def __init__(self, layers):
        self.layers = layers

    def forward_propagate(self, features):
        outputs = []
        next_input = features
        for layer in self.layers:
            output = [neuron.activate(next_input) for neuron in layer]
            outputs.append(output)
            next_input = output
        return outputs
