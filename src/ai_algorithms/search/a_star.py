from ai_algorithms.search.graph_search import graph_search
from search.candidates.priority_queue import CandidatePriorityQueue
from search.strategy import Strategy


class AStar(Strategy):

    def __init__(self, model, heuristic):
        self.model = model
        self.heuristic = heuristic

    def create_candidates(self):
        return CandidatePriorityQueue()

    def add_to_candidates(self, node, candidates):
        candidates.add_candidate(node, node.cost + self.heuristic(self.model, node.state))


def a_star_search(model, heuristic):
    return graph_search(model, AStar(model, heuristic))
