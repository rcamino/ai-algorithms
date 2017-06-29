from ai_algorithms.search.graph_search import graph_search
from ai_algorithms.search.candidates.priority_queue import CandidatePriorityQueue
from ai_algorithms.search.strategy import Strategy


class UCS(Strategy):
    """
    Uniform Cost Search strategy.
    Uses a priority queue for the candidates, using the cost of the state as priority value.
    Hence, candidates with lower costs are visited first.
    """

    def create_candidates(self):
        return CandidatePriorityQueue()

    def add_to_candidates(self, node, candidates):
        candidates.add_candidate(node, node.cost)


# using always the same instance because it is stateless
ucs_strategy = UCS()


def ucs_search(model):
    """
    Shorthand for graph search using the Uniform Cost Search strategy.
    :param model: must implement ai_algorithms.model.{Model,Deterministic,CostAware}
    :return: action sequence or None if the problem cannot be solved
    """
    return graph_search(model, ucs_strategy)
