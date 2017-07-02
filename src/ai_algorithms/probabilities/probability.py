def normalize(denormalized):
    """
    Normalizes a distribution.
    :param denormalized: dictionary of probabilities for every value of a random variable
    :return: dictionary of probabilities for every value of a random variable summing to one
    """
    total = float(sum(denormalized.values()))
    return dict([(value, p / total) for value, p in denormalized.items()])
