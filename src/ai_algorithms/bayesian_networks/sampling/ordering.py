import heapq
import random


def random_topological_order(root_nodes, random_state=None):
    """
    Generates an ordering for nodes from a Bayesian Network following the links them starting from the root nodes,
    breaking ties in with random ordering.
    :param root_nodes: Bayesian Network; collection of ai_algorithms.bayesian_networks.node.BayesianNetworkNode
    :param random_state: random.RandomState; if None, default random state will be used
    :return: sequence of ai_algorithms.bayesian_networks.node.BayesianNetworkNode
    """
    if random_state is None:
        random_state = random.Random()
    nodes = []
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
