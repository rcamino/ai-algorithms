from ai_algorithms.search.candidates.priority_queue import CandidatePriorityQueue
from ai_algorithms.search.graph_search import graph_search
from ai_algorithms.search.strategy import Strategy


class AStar(Strategy):
    """
    A* strategy.
    Uses a priority queue for the candidates, using (cost + estimated future cost) as priority value.
    Hence, candidates with lower (cost + estimated future cost) are visited first.
    """

    def __init__(self, model, heuristic):
        self.model = model
        self.heuristic = heuristic

    def create_candidates(self):
        return CandidatePriorityQueue()

    def add_to_candidates(self, node, candidates):
        candidates.add_candidate(node, node.cost + self.heuristic(self.model, node.state))


def a_star_search(model, heuristic):
    """
    Shorthand for graph search using the A* strategy.
    :param model: must implement ai_algorithms.model.{Model,Deterministic,CostAware}
    :param heuristic: a function that estimates the cost given the model and a state
    :return: action sequence or None if the problem cannot be solved
    """
    return graph_search(model, AStar(model, heuristic))
