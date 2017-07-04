from ai_algorithms.bayesian_networks.sampling.ordering import random_topological_order
from ai_algorithms.probabilities.inference import observe
from ai_algorithms.probabilities.sampling import sample_from_distribution


def prior_sampling(root_nodes, random_state=None):
    """
    Generates a sample from a Bayesian Network following the links between probability tables.
    For nodes in the same level, ties are broken in random order.
    :param root_nodes: Bayesian Network; collection of ai_algorithms.bayesian_networks.node.BayesianNetworkNode
    :param random_state: random.RandomState; if None, default random state will be used
    :return: sample; dictionary of variable names to values
    """
    nodes = random_topological_order(root_nodes, random_state)
    observations_by_name = {}
    for node in nodes:
        observations_by_name[node.name] = prior_sampling_step(node, observations_by_name, random_state)
    return observations_by_name


def prior_sampling_step(node, observations_by_name, random_state):
    """
    Generates a sample from a Bayesian Network node probability table given observations from all of its parents.
    :param node: ai_algorithms.bayesian_networks.node.BayesianNetworkNode
    :param observations_by_name: observations from parent nodes; dictionary of node name to value
    :param random_state: random.RandomState; if None, default random state will be used
    :return: sample value from the node
    """
    evidence = node.probability.create_evidence(observations_by_name)
    probability = observe(node.probability, evidence)
    return sample_from_distribution(probability, random_state)
