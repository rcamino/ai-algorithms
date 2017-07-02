from ai_algorithms.bayesian_networks.sampling.ordering import random_topological_order
from ai_algorithms.bayesian_networks.sampling.prior import prior_sampling_step
from ai_algorithms.probabilities.inference import observe
from ai_algorithms.probabilities.sampling import sample_from_distribution


def likelihood_weighting(root_nodes, observations_by_name, random_state=None):
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
