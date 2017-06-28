from search.candidates.priority_queue import CandidatePriorityQueue
from search.graph_search import graph_search
from search.strategy import Strategy


class AStar(Strategy):

    def __init__(self, search_problem, heuristic):
        self.search_problem = search_problem
        self.heuristic = heuristic

    def create_candidates(self):
        return CandidatePriorityQueue()

    def add_to_candidates(self, node, candidates):
        candidates.add_candidate(node, node.cost + self.heuristic(self.search_problem, node.state))


def a_star_search(search_problem, heuristic):
    return graph_search(search_problem, AStar(search_problem, heuristic))
