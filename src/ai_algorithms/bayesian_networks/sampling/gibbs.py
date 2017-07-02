import random

from ai_algorithms.bayesian_networks.sampling.ordering import random_topological_order
from ai_algorithms.bayesian_networks.sampling.prior import prior_sampling_step


def gibbs_sampling(root_nodes, observations_by_name, iterations=1000, random_state=None):
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
