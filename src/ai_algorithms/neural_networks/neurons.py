class Neuron(object):

    def __init__(self, weights, activation_function):
        self.weights = weights
        self.activation_function = activation_function

    def activate(self, inputs):
        output = 0.0
        for input_i, weight_i in zip(inputs, self.weights):
            output += input_i * weight_i
        return self.activation_function.activate(output)
