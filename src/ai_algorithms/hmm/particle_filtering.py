from ai_algorithms.probabilities.probability import sample_from_distribution, samples_from_distribution


def transition_sample(particles, transition_probability, random_state=None):
    """
    Randomly moves the particles according to transition probabilities.
    :param particles: list of states; they must be hashable objects
    :param transition_probability: dictionary of dictionaries of probabilities for states given previous states
    :param random_state: random.RandomState; if None, default random state will be used
    :return: list of states; they must be hashable objects
    """
    return [sample_from_distribution(transition_probability[particle], random_state) for particle in particles]


def resample(particles, emission_probability, observation, random_state=None):
    """
    Creates a new particles based on weights given by emission probabilities and particles per state.
    :param particles: list of states; they must be hashable objects
    :param emission_probability: dictionary of dictionaries of probabilities for observations given states
    :param observation: must be a hashable object
    :param random_state: random.RandomState; if None, default random state will be used
    :return: list of states; they must be hashable objects
    """
    particle_weights = [emission_probability[observation][particle] for particle in particles]
    total_weight = float(sum(particle_weights))
    normalized_weights = [weight / total_weight for weight in particle_weights]
    probability = {}
    for particle, normalized_weight in zip(particles, normalized_weights):
        if particle not in probability:
            probability[particle] = 0.0
        probability[particle] += normalized_weight
    return [sample_from_distribution(probability, random_state) for _ in particles]


def probability_from_particles(states, particles):
    """
    Calculates the probability per state given the particles.
    :param states: list of states; they must be hashable objects
    :param particles: list of states; they must be hashable objects
    :return: dictionary of probabilities for every hidden state
    """
    p = dict([(state, 0.0) for state in states])
    for particle in particles:
        p[particle] += 1.0
    total = float(len(particles))
    for state in states:
        p[state] /= total
    return p


def particle_filtering(transition_probability, emission_probability, observations, prior, size, random_state=None):
    """
    Calculates the probability of every hidden state after a sequence of time steps and observations.
    :param transition_probability: dictionary of dictionaries of probabilities for states given previous states
    :param emission_probability: dictionary of dictionaries of probabilities for observations given states
    :param observations: list of observations; they must be hashable objects
    :param prior: dictionary of probabilities for the initial probability of every hidden state
    :param size: number of particles to use
    :param random_state: random.RandomState; if None, default random state will be used
    :return: dictionary of probabilities for every hidden state
    """
    particles = samples_from_distribution(prior, size, random_state)
    for observation in observations:
        particles = transition_sample(particles, transition_probability, random_state)
        particles = resample(particles, emission_probability, observation, random_state)
    return probability_from_particles(prior.keys(), particles)
