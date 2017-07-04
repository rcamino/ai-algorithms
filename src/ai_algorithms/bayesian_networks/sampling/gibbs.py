import random

from ai_algorithms.bayesian_networks.sampling.ordering import random_topological_order
from ai_algorithms.bayesian_networks.sampling.prior import prior_sampling_step


def gibbs_sampling(root_nodes, observations_by_name, iterations=1000, random_state=None):
    """
    Generates a sample from a Bayesian Network.
    First the existing observations are fixed in the sample.
    Then random valid values are assigned to the rest of the nodes.
    Then an iterative process starts, and in every iteration,
    non fixed node is chosen to re-sampled its value using the other nodes as observations.
    :param root_nodes: Bayesian Network; collection of ai_algorithms.bayesian_networks.node.BayesianNetworkNode
    :param observations_by_name: nodes that were already observed; dictionary of node name to value
    :param iterations: maximum number of iterations
    :param random_state: random.RandomState; if None, default random state will be used
    :return: sample; dictionary of variable names to values
    """
    if random_state is None:
        random_state = random.Random()

    new_observations_by_name = {}
    non_evidence_nodes = []
    nodes = random_topological_order(root_nodes, random_state)
    for node in nodes:
        if node.name in observations_by_name:
            # fix observations
            new_observations_by_name[node.name] = observations_by_name[node.name]
        else:
            # random initialisation
            values = node.probability.values_by_name[node.name]
            new_observations_by_name[node.name] = random_state.choice(values)

            non_evidence_nodes.append(node)

    iteration = 0
    while iteration < iterations:
        node = random.choice(non_evidence_nodes)
        new_observations_by_name.pop(node.name)
        new_observations_by_name[node.name] = prior_sampling_step(node, new_observations_by_name, random_state)
        iteration += 1
    return new_observations_by_name
