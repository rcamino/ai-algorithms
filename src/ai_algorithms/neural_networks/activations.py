import math


class ActivationUnit(object):

    def activate(self, inputs):
        raise NotImplementedError


class BinaryUnit(ActivationUnit):

    def activate(self, inputs):
        return [1.0 if value > 0.0 else 0.0 for value in inputs]


class LinearUnit(ActivationUnit):

    def __init__(self, weight_matrix):
        self.weight_matrix = weight_matrix

    def activate(self, inputs):
        outputs = []
        for weights in self.weight_matrix:
            output_i = 0.0
            for input_i, weight_i in zip(inputs, weights):
                output_i += input_i * weight_i
            outputs.append(output_i)
        return outputs


class RectifiedUnit(ActivationUnit):

    def activate(self, inputs):
        return [input_i if input_i > 0.0 else 0.0 for input_i in inputs]


class Sigmoid(ActivationUnit):

    def activate(self, inputs):
        return [1.0 / (1.0 + math.exp(-input_i)) for input_i in inputs]


class Softmax(ActivationUnit):

    def __init__(self):
        self.sigmoid = Sigmoid()

    def activate(self, inputs):
        sigmoid_outputs = self.sigmoid.activate(inputs)
        norm = sum(sigmoid_outputs)
        return [sigmoid_i / norm for sigmoid_i in sigmoid_outputs]


class HyperbolicTangent(ActivationUnit):

    def activate(self, inputs):
        return map(math.tanh, inputs)
