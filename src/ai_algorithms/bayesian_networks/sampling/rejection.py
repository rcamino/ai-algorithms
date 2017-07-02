from ai_algorithms.bayesian_networks.sampling.ordering import random_topological_order
from ai_algorithms.probabilities.inference import observe
from ai_algorithms.probabilities.sampling import sample_from_distribution


def rejection_sampling(root_nodes, observations_by_name, iterations=1000, random_state=None):
    repeat = True
    iteration = 0
    while repeat and iteration < iterations:
        repeat = False
        nodes = random_topological_order(root_nodes, random_state)
        value_by_name = {}
        for node in nodes:
            node_observations_by_name = {}
            for parent in node.parents:
                node_observations_by_name[parent.name] = value_by_name[parent.name]
            evidence = node.probability.create_evidence(node_observations_by_name)
            probability = observe(node.probability, evidence)
            value = sample_from_distribution(probability, random_state)
            if (node.name not in observations_by_name) or (observations_by_name[node.name] == value):
                value_by_name[node.name] = value
            else:
                repeat = True
                iteration += 1
                break
    if iteration >= iterations:
        raise Exception("Iteration limit reached.")
    return value_by_name
