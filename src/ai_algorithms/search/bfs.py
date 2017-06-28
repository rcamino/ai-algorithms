from ai_algorithms.search.graph_search import graph_search
from ai_algorithms.search.candidates.queue import CandidateQueue
from ai_algorithms.search.strategy import Strategy


class BFS(Strategy):

    def create_candidates(self):
        return CandidateQueue()

    def add_to_candidates(self, node, candidates):
        candidates.add_candidate(node)


bfs_strategy = BFS()


def bfs_search(model):
    return graph_search(model, bfs_strategy)
