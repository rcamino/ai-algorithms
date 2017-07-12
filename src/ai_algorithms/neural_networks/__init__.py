import random


def add_bias_to_features(features):
    return [1.0] + features


def add_bias_to_samples(samples):
    return map(add_bias_to_features, samples)


def create_random_weights(size, random_state=None):
    if random_state is None:
        random_state = random.Random()

    weights = []
    for _ in xrange(size):
        weights.append(random_state.random())
    return weights
