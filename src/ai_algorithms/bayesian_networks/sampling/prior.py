from ai_algorithms.bayesian_networks.sampling.ordering import random_topological_order
from ai_algorithms.probabilities.inference import observe
from ai_algorithms.probabilities.sampling import sample_from_distribution


def prior_sampling(root_nodes, random_state=None):
    nodes = random_topological_order(root_nodes, random_state)
    value_by_name = {}
    for node in nodes:
        observations_by_name = {}
        for parent in node.parents:
            observations_by_name[parent.name] = value_by_name[parent.name]
        evidence = node.probability.create_evidence(observations_by_name)
        probability = observe(node.probability, evidence)
        value = sample_from_distribution(probability, random_state)
        value_by_name[node.name] = value
    return value_by_name
