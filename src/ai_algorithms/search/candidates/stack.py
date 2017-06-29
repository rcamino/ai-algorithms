from collections import deque

from ai_algorithms.search.candidates import Candidates


class CandidateStack(Candidates):

    def __init__(self):
        self.candidates = deque()

    def has_candidates(self):
        """
        :return: True if have more candidates, False otherwise
        """
        return len(self.candidates) > 0

    def next_candidate(self):
        """
        Removes the next candidate node from the top of the stack and returns it.
        :return: ai_algorithms.search.graph_search.Node
        """
        return self.candidates.pop()

    def add_candidate(self, candidate):
        """
        Add the candidate to the top of the stack.
        :param candidate: ai_algorithms.search.graph_search.Node
        """
        self.candidates.append(candidate)
