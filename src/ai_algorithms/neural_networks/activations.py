import math


class ActivationUnit(object):

    def activate(self, values):
        raise NotImplementedError


class LinearUnit(ActivationUnit):

    def activate(self, values):
        return values


class RectifiedLinearUnit(ActivationUnit):

    def activate(self, values):
        return [value if value > 0.0 else 0.0 for value in values]


class Sigmoid(ActivationUnit):

    def activate(self, values):
        return [1.0 / (1.0 + math.exp(-value)) for value in values]


class Softmax(ActivationUnit):

    def activate(self, values):
        denominator = sigmoid.activate(values)
        numerator = sum(denominator)
        return [denominator_i / numerator for denominator_i in denominator]


class HyperbolicTangent(ActivationUnit):

    def activate(self, values):
        return map(math.tanh, values)


linear = LinearUnit()
relu = RectifiedLinearUnit()
sigmoid = Sigmoid()
softmax = Softmax()
tanh = HyperbolicTangent()
