def mini_forward_step(transition_probability, previous):
    """
    Calculates the probability distribution for every state in a Markov chain given the previous state distribution.
    :param transition_probability: probability of moving from one state to another; dictionary of dictionaries of floats
    :param previous: dictionary of probabilities for every state of the previous distribution
    :return: dictionary of probabilities for every state
    """
    p = {}
    for x in previous.keys():
        p[x] = 0.0
        for x_previous, p_previous in previous.items():
            p[x] += transition_probability[x_previous][x] * p_previous
    return p


def mini_forward(transition_probability, prior, iterations=1000, tolerance=1e-6):
    """
    Calculates the stationary probability distribution for every state in a Markov chain.
    :param transition_probability: probability of moving from one state to another; dictionary of dictionaries of floats
    :param prior: dictionary of probabilities for every state of the initial distribution
    :param iterations: maximum number of iterations
    :param tolerance: if the maximum difference between all the values is less than this number the algorithm stops
    :return: dictionary of probabilities for every state
    """
    p = prior
    for iteration in xrange(iterations):
        p_next = mini_forward_step(transition_probability, p)
        max_change = 0.0
        for x in p.keys():
            max_change = max(max_change, abs(p_next[x] - p[x]))
        if max_change < tolerance:
            break
        p = p_next
    return p
