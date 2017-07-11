import random


def add_bias_to_features(features):
    return [1.0] + features


def add_bias_to_samples(samples):
    return map(add_bias_to_features, samples)


def create_random_weight_matrix(input_size, output_size, random_state=None):
    return [create_random_weights(input_size, random_state) for _ in xrange(output_size)]


def create_random_weights(size, random_state=None):
    if random_state is None:
        random_state = random.Random()

    weights = []
    for _ in xrange(size):
        weights.append(random_state.random())
    return weights
