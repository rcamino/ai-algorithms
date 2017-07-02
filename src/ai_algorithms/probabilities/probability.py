import random


def bernoulli_trial(p=0.5, random_state=None):
    """
    Performs a Bernoulli trial with a given probability of success.
    :param p: optional probability of success; defaults to 0.5
    :param random_state: random.RandomState; if None, default random state will be used
    :return: True when success, False otherwise
    """
    if random_state is None:
        random_state = random.Random()
    return random_state.random() < p


def samples_from_distribution(probability, size, random_state=None):
    """
    Draws values from a random variable given its distribution.
    :param probability: dictionary of probabilities for every value of a random variable
    :param size: number of particles to generate
    :param random_state: random.RandomState; if None, default random state will be used
    :return: list of values; they must be hashable objects
    """
    return [sample_from_distribution(probability, random_state) for _ in xrange(size)]


def sample_from_distribution(probability, random_state=None):
    """
    Draws a value from a random variable given its distribution.
    :param probability: dictionary of probabilities for every value of a random variable
    :param random_state: random.RandomState; if None, default random state will be used
    :return: value from a random variable; must be a hashable object
    """
    if random_state is None:
        random_state = random.Random()
    values = sorted(probability.keys())
    r = random_state.random()
    accumulated = 0.0
    for value in values:
        accumulated += probability[value]
        if r < accumulated:
            return value
    raise Exception("Invalid distribution.")
