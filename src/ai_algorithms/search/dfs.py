from ai_algorithms.search.candidates.stack import CandidateStack
from ai_algorithms.search.graph_search import graph_search
from ai_algorithms.search.strategy import Strategy


class DFS(Strategy):

    def create_candidates(self):
        return CandidateStack()

    def add_to_candidates(self, node, candidates):
        candidates.add_candidate(node)


dfs_strategy = DFS()


def dfs_search(model):
    return graph_search(model, dfs_strategy)
