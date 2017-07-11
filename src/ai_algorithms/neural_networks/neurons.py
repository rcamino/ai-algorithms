import math


class Neuron(object):

    def activate(self, inputs):
        raise NotImplementedError


class Binary(Neuron):

    def activate(self, inputs):
        return [1.0 if value > 0.0 else 0.0 for value in inputs]


class Linear(Neuron):

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


class Rectified(Neuron):

    def activate(self, inputs):
        return [input_i if input_i > 0.0 else 0.0 for input_i in inputs]


class Logistic(Neuron):

    def activate(self, inputs):
        return [1.0 / (1.0 + math.exp(-input_i)) for input_i in inputs]


class Softmax(Neuron):

    def __init__(self):
        self.logistic = Logistic()

    def activate(self, inputs):
        logits = self.logistic.activate(inputs)
        norm = sum(logits)
        return [logit / norm for logit in logits]


class HyperbolicTangent(Neuron):

    def activate(self, inputs):
        return map(math.tanh, inputs)
