def mini_forward_step(conditional_distribution, previous):
    """
    Calculates the probability distribution for every state in a Markov chain given the previous state distribution.
    :param conditional_distribution: dictionary of dictionaries of probabilities for states given previous states
    :param previous: dictionary of probabilities for every state of the previous distribution
    :return: dictionary of probabilities for every state
    """
    p = {}
    for x in previous.keys():
        p[x] = 0.0
        for x_previous, p_previous in previous.items():
            p[x] += conditional_distribution[x][x_previous] * p_previous
    return p


def mini_forward(conditional_distribution, prior, iterations=1000, tolerance=1e-6):
    """
    Calculates the stationary probability distribution for every state in a Markov chain.
    :param conditional_distribution: dictionary of dictionaries of probabilities for states given previous states
    :param prior: dictionary of probabilities for every state of the initial distribution
    :param iterations: maximum number of iterations
    :param tolerance: if the maximum difference between all the values is less than this number the algorithm stops
    :return: dictionary of probabilities for every state
    """
    p = prior
    for iteration in xrange(iterations):
        p_next = mini_forward_step(conditional_distribution, p)
        max_change = 0.0
        for x in p.keys():
            max_change = max(max_change, abs(p_next[x] - p[x]))
        if max_change < tolerance:
            break
        p = p_next
    return p
