from search.candidates.queue import CandidateQueue
from search.graph_search import graph_search
from search.strategy import Strategy


class BFS(Strategy):

    def create_candidates(self):
        return CandidateQueue()

    def add_to_candidates(self, node, candidates):
        candidates.add_candidate(node)


bfs_strategy = BFS()


def bfs_search(search_problem):
    return graph_search(search_problem, bfs_strategy)
