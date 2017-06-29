from ai_algorithms.search.candidates.stack import CandidateStack
from ai_algorithms.search.graph_search import graph_search
from ai_algorithms.search.strategy import Strategy


class DFS(Strategy):
    """
    Depth First Search strategy.
    Uses a stack for the candidates.
    This causes the graph search algorithm to traverse directly to the leaves first
    Costs are ignored.
    """

    def create_candidates(self):
        return CandidateStack()

    def add_to_candidates(self, node, candidates):
        candidates.add_candidate(node)


# using always the same instance because it is stateless
dfs_strategy = DFS()


def dfs_search(model):
    """
    Shorthand for graph search using the Depth First Search strategy.
    :param model: must implement ai_algorithms.model.{Model,Deterministic,CostAware}
    :return: action sequence or None if the problem cannot be solved
    """
    return graph_search(model, dfs_strategy)
