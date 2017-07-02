from ai_algorithms.bayesian_networks.sampling.ordering import random_topological_order
from ai_algorithms.bayesian_networks.sampling.prior import prior_sampling_step


def rejection_sampling(root_nodes, observations_by_name, iterations=1000, random_state=None):
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
