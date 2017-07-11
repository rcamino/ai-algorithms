import random


class FeedForwardLayer(object):

    def __init__(self, weight_matrix, activation):
        self.weight_matrix = weight_matrix
        self.activation = activation

    def score(self, features):
        results = []
        for weights in self.weight_matrix:
            result = 0.0
            for feature, weight in zip(features, weights):
                result += feature * weight
            results.append(result)
        return self.activation.activate(results)


class FeedForwardNetwork(object):

    def __init__(self, layers):
        self.layers = layers

    def score(self, features):
        values = features
        for layer in self.layers:
            values = layer.score(values)
        return values

    def predict(self, features):
        pairs = enumerate(self.score(features))
        sorted_pairs = sorted(pairs, reverse=True, key=lambda pair: pair[1])
        best_pair = sorted_pairs[0]
        return best_pair[0]


def create_random_weight_matrix(input_size, output_size, random_state=None):
    return [create_random_weights(input_size, random_state) for _ in xrange(output_size)]


def create_random_weights(size, random_state=None):
    if random_state is None:
        random_state = random.Random()

    weights = []
    for _ in xrange(size):
        weights.append(random_state.random())
    return weights
