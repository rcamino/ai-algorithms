from ai_algorithms.markov_chains.mini_forward import mini_forward_step


def observation_step(emission_probability, observation, probability):
    """
    Calculates the probability of every hidden state after an observation.
    :param emission_probability: dictionary of dictionaries of probabilities for observations given states
    :param observation: must be a hashable object
    :param probability: dictionary of probabilities for every hidden state in the current time step
    :return: dictionary of probabilities for every hidden state
    """
    p = {}
    for x, p_x in probability.items():
        p[x] = p_x * emission_probability[observation][x]
    return p


def forward_step(transition_probability, emission_probability, observation, previous):
    """
    Calculates the probability of every hidden state after a time step and an observation.
    :param transition_probability: dictionary of dictionaries of probabilities for states given previous states
    :param emission_probability: dictionary of dictionaries of probabilities for observations given states
    :param observation: must be a hashable object
    :param previous: dictionary of probabilities for every hidden state in the previous time step
    :return: dictionary of probabilities for every hidden state
    """
    p = mini_forward_step(transition_probability, previous)
    return observation_step(emission_probability, observation, p)


def forward(transition_probability, emission_probability, observations, prior):
    """
    Calculates the probability of every hidden state after a sequence of time steps and observations.
    :param transition_probability: dictionary of dictionaries of probabilities for states given previous states
    :param emission_probability: dictionary of dictionaries of probabilities for observations given states
    :param observations: list of observations; they must be hashable objects
    :param prior: dictionary of probabilities for the initial probability of every hidden state
    :return: dictionary of probabilities for every hidden state
    """
    p = prior
    for observation in observations:
        p = forward_step(transition_probability, emission_probability, observation, p)
    return p
