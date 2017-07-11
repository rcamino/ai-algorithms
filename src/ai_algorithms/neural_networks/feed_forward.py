import random

class FeedForwardNetwork(object):

    def __init__(self, layers):
        self.layers = layers

    def score(self, features):
        last_layer_output = features
        for layer in self.layers:
            last_layer_output = layer.activate(last_layer_output)
        return last_layer_output

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
