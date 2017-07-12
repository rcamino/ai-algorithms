class Neuron(object):

    def __init__(self, weights, activation):
        self.weights = weights
        self.activation = activation

    def activate(self, inputs):
        output = 0.0
        for input_i, weight_i in zip(inputs, self.weights):
            output += input_i * weight_i
        return self.activation.activate(output)
