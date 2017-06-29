import heapq

from ai_algorithms.search.candidates import Candidates


class CandidatePriorityQueue(Candidates):

    def __init__(self):
        self.candidates = []
        self.count = 0

    def has_candidates(self):
        return len(self.candidates) > 0

    def next_candidate(self):
        priority, count, candidate = heapq.heappop(self.candidates)
        return candidate

    def add_candidate(self, candidate, cost):
        order = self.count
        self.count += 1
        heapq.heappush(self.candidates, (cost, order, candidate))
