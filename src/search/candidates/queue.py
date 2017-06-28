from collections import deque

from search.candidates import Candidates


class CandidateQueue(Candidates):

    def __init__(self):
        self.candidates = deque()

    def has_candidates(self):
        return len(self.candidates) > 0

    def next_candidate(self):
        return self.candidates.popleft()

    def add_candidate(self, candidate):
        self.candidates.append(candidate)
