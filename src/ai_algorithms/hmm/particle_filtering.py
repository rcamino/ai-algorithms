from ai_algorithms.probabilities.inference import observe
from ai_algorithms.probabilities.probability import normalize
from ai_algorithms.probabilities.sampling import sample_from_distribution, samples_from_distribution


def transition_sample(particles, transition_probability, random_state=None):
    """
    Randomly moves the particles according to transition probabilities.
    :param particles: list of states; they must be hashable objects
    :param transition_probability: dictionary of dictionaries of probabilities for states given previous states
    :param random_state: random.RandomState; if None, default random state will be used
    :return: list of states; they must be hashable objects
    """
    return [sample_from_distribution(observe(transition_probability, [None, particle]), random_state)
            for particle in particles]


def resample(particles, emission_probability, observation, random_state=None):
    """
    Creates a new particles based on weights given by emission probabilities and particles per state.
    :param particles: list of states; they must be hashable objects
    :param emission_probability: dictionary of dictionaries of probabilities for observations given states
    :param observation: must be a hashable object
    :param random_state: random.RandomState; if None, default random state will be used
    :return: list of states; they must be hashable objects
    """
    probability = {}
    for particle in particles:
        if particle not in probability:
            probability[particle] = 0.0
        probability[particle] += emission_probability[observation][particle]
    probability = normalize(probability)
    return [sample_from_distribution(probability, random_state) for _ in particles]


def probability_from_particles(states, particles):
    """
    Calculates the probability per state given the particles.
    :param states: list of states; they must be hashable objects
    :param particles: list of states; they must be hashable objects
    :return: dictionary of probabilities for every hidden state
    """
    probability = dict([(state, 0.0) for state in states])
    for particle in particles:
        probability[particle] += 1.0
    return normalize(probability)


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
