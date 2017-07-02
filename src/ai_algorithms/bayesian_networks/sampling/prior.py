from ai_algorithms.bayesian_networks.sampling.ordering import random_topological_order
from ai_algorithms.probabilities.inference import observe
from ai_algorithms.probabilities.sampling import sample_from_distribution


def prior_sampling(root_nodes, random_state=None):
    nodes = random_topological_order(root_nodes, random_state)
    observations_by_name = {}
    for node in nodes:
        observations_by_name[node.name] = prior_sampling_step(node, observations_by_name, random_state)
    return observations_by_name


def prior_sampling_step(node, observations_by_name, random_state):
    evidence = node.probability.create_evidence(observations_by_name)
    probability = observe(node.probability, evidence)
    return sample_from_distribution(probability, random_state)
