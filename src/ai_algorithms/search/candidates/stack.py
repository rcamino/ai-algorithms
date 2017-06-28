from collections import deque

from ai_algorithms.search.candidates import Candidates


class CandidateStack(Candidates):

    def __init__(self):
        self.candidates = deque()

    def has_candidates(self):
        return len(self.candidates) > 0

    def next_candidate(self):
        return self.candidates.pop()

    def add_candidate(self, candidate):
        self.candidates.append(candidate)
