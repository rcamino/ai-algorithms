from ai_algorithms.search.graph_search import graph_search
from ai_algorithms.search.candidates.priority_queue import CandidatePriorityQueue
from ai_algorithms.search.strategy import Strategy


class UCS(Strategy):
    def create_candidates(self):
        return CandidatePriorityQueue()

    def add_to_candidates(self, node, candidates):
        candidates.add_candidate(node, node.cost)


ucs_strategy = UCS()


def ucs_search(model):
    return graph_search(model, ucs_strategy)