import heapq
import random

from ai_algorithms.probabilities.inference import observe
from ai_algorithms.probabilities.sampling import sample_from_distribution


def random_topological_order(root_nodes, random_state=None):
    nodes = []
    if random_state is None:
        random_state = random.Random()
    priority_queue = []
    for root_node in root_nodes:
        # priority is root level = 0
        # break ties with random number
        heapq.heappush(priority_queue, (0, random_state.random(), root_node))
    visited = set()
    while len(priority_queue) > 0:
        level, _, node = heapq.heappop(priority_queue)
        if node not in visited:
            visited.add(node)
            nodes.append(node)
            for child in node.children:
                # priority is parent level + 1
                # break ties with random number
                heapq.heappush(priority_queue, (level + 1, random_state.random(), child))
    return nodes


def prior_sampling(root_nodes, random_state=None):
    if random_state is None:
        random_state = random.Random()
    nodes = random_topological_order(root_nodes, random_state)
    values = []
    value_by_node = {}
    for node in nodes:
        observations_by_name = {}
        for parent in node.parents:
            observations_by_name[parent.name] = value_by_node[parent]
        evidence = node.probability.create_evidence(observations_by_name)
        probability = observe(node.probability, evidence)
        value = sample_from_distribution(probability, random_state)
        values.append(value)
        value_by_node[node] = value
    return values
