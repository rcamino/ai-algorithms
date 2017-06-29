from ai_algorithms.search.graph_search import graph_search
from ai_algorithms.search.candidates.queue import CandidateQueue
from ai_algorithms.search.strategy import Strategy


class BFS(Strategy):
    """
    Breath First Search strategy.
    Uses a queue for the candidates.
    This causes the graph search algorithm to traverse by levels.
    Costs are ignored.
    """

    def create_candidates(self):
        return CandidateQueue()

    def add_to_candidates(self, node, candidates):
        candidates.add_candidate(node)


# using always the same instance because it is stateless
bfs_strategy = BFS()


def bfs_search(model):
    """
    Shorthand for graph search using the Breath First Search strategy.
    :param model: must implement ai_algorithms.model.{Model,Deterministic,CostAware}
    :return: action sequence or None if the problem cannot be solved
    """
    return graph_search(model, bfs_strategy)
