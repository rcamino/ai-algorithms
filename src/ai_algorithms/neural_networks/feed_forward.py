import random


class FeedForwardLayer(object):

    def __init__(self, weight_matrix, activation):
        self.weight_matrix = weight_matrix
        self.activation = activation

    def emit(self, inputs):
        outputs = []
        for weights in self.weight_matrix:
            output_i = 0.0
            for input_i, weight_i in zip(inputs, weights):
                output_i += input_i * weight_i
            outputs.append(output_i)
        return self.activation.activate(outputs)


class FeedForwardNetwork(object):

    def __init__(self, layers):
        self.layers = layers

    def score(self, features):
        values = features
        for layer in self.layers:
            values = layer.emit(values)
        return values

    def predict(self, features):
        scores = self.score(features)
        predictions = range(len(scores))
        return sorted(predictions, reverse=True, key=lambda prediction: scores[prediction])[0]


def create_random_weight_matrix(input_size, output_size, random_state=None):
    return [create_random_weights(input_size, random_state) for _ in xrange(output_size)]


def create_random_weights(size, random_state=None):
    if random_state is None:
        random_state = random.Random()

    weights = []
    for _ in xrange(size):
        weights.append(random_state.random())
    return weights
