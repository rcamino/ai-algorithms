from ai_algorithms.bayesian_networks.sampling.ordering import random_topological_order
from ai_algorithms.bayesian_networks.sampling.prior import prior_sampling_step


def rejection_sampling(root_nodes, observations_by_name, iterations=1000, random_state=None):
    """
    Generates a sample from a Bayesian Network following the links between probability tables.
    For nodes in the same level, ties are broken in random order.
    When there is an observation for a node and the sample contradicts that observation,
    the process is repeated from the beginning, up to a number of maximum iterations.
    :param root_nodes: Bayesian Network; collection of ai_algorithms.bayesian_networks.node.BayesianNetworkNode
    :param observations_by_name: nodes that were already observed; dictionary of node name to value
    :param iterations: maximum number of iterations
    :param random_state: random.RandomState; if None, default random state will be used
    :return: sample; dictionary of variable names to values
    """
    repeat = True
    iteration = 0
    while repeat and iteration < iterations:
        repeat = False
        nodes = random_topological_order(root_nodes, random_state)
        new_observations_by_name = {}
        for node in nodes:
            value = prior_sampling_step(node, new_observations_by_name, random_state)
            if (node.name not in observations_by_name) or (observations_by_name[node.name] == value):
                new_observations_by_name[node.name] = value
            else:
                repeat = True
                iteration += 1
                break
    if iteration >= iterations:
        raise Exception("Iteration limit reached.")
    return new_observations_by_name
