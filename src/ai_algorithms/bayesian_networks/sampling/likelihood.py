from ai_algorithms.bayesian_networks.sampling.ordering import random_topological_order
from ai_algorithms.bayesian_networks.sampling.prior import prior_sampling_step
from ai_algorithms.probabilities.inference import observe
from ai_algorithms.probabilities.sampling import sample_from_distribution


def likelihood_weighting(root_nodes, observations_by_name, random_state=None):
    """
    Generates a sample from a Bayesian Network following the links between probability tables.
    For nodes in the same level, ties are broken in random order.
    The sample is generated with a weight, starting from 1.
    When there is an observation for a node, the observation is used instead of picking a random value,
    and the weight is multiplied by the probability of that sample given the parent observations.
    :param root_nodes: Bayesian Network; collection of ai_algorithms.bayesian_networks.node.BayesianNetworkNode
    :param observations_by_name: nodes that were already observed; dictionary of node name to value
    :param random_state: random.RandomState; if None, default random state will be used
    :return: sample; dictionary of variable names to values
    """
    nodes = random_topological_order(root_nodes, random_state)
    new_observations_by_name = {}
    weight = 1.0
    for node in nodes:
        if node.name in observations_by_name:
            new_observations_by_name[node.name] = observations_by_name[node.name]

            parents_observations_by_name = {node.name: observations_by_name[node.name]}
            for parent in node.parents:
                parents_observations_by_name[parent.name] = new_observations_by_name[parent.name]
            parent_evidence = node.probability.create_evidence(parents_observations_by_name)
            weight *= observe(node.probability, parent_evidence)
        else:
            new_observations_by_name[node.name] = prior_sampling_step(node, new_observations_by_name, random_state)
    return new_observations_by_name, weight


def probability_from_weighted_samples(positive_weights, all_weights):
    """
    Calculates the probability of a sample given the weights for the sample and the weights of all the samples.
    The first collection must be included in the second one.
    :param positive_weights: list of floats
    :param all_weights: list of floats
    :return: float between [0, 1]
    """
    if len(positive_weights) == 0:
        return None

    if len(positive_weights) == 0:
        return 0.0
    else:
        return sum(positive_weights) / float(sum(all_weights))
