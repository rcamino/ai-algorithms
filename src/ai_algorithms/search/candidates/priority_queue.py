import heapq

from ai_algorithms.search.candidates import Candidates


class CandidatePriorityQueue(Candidates):

    def __init__(self):
        self.candidates = []
        self.count = 0

    def has_candidates(self):
        """
        :return: True if have more candidates, False otherwise
        """
        return len(self.candidates) > 0

    def next_candidate(self):
        """
        Removes the next candidate node with the lowest cost from the structure and returns it.
        FIFO is used as tie breaker.
        :return: ai_algorithms.search.graph_search.Node
        """
        _, _, candidate = heapq.heappop(self.candidates)
        return candidate

    def add_candidate(self, candidate, cost):
        """
        Add the candidate giving more priority to the lower costs.
        :param candidate: ai_algorithms.search.graph_search.Node
        :param cost: float or int
        """
        order = self.count
        self.count += 1
        heapq.heappush(self.candidates, (cost, order, candidate))
