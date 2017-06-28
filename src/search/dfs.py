from search.candidates.stack import CandidateStack
from search.graph_search import graph_search
from search.strategy import Strategy


class DFS(Strategy):

    def create_candidates(self):
        return CandidateStack()

    def add_to_candidates(self, node, candidates):
        candidates.add_candidate(node)


dfs_strategy = DFS()


def dfs_search(search_problem):
    return graph_search(search_problem, dfs_strategy)
