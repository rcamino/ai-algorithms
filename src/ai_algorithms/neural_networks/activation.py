import math


class Activation(object):

    def activate(self, neuron_output):
        raise NotImplementedError

    def error_derivative(self, activation):
        raise NotImplementedError


class Logistic(Activation):

    def activate(self, neuron_output):
        return 1.0 / (1.0 + math.exp(-neuron_output))

    def error_derivative(self, activation):
        return activation * (1.0 - activation)
