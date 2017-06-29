class Candidates(object):
    """
    Interface for the candidate nodes structure.
    """

    def has_candidates(self):
        """
        :return: True if have more candidates, False otherwise
        """
        raise NotImplementedError

    def next_candidate(self):
        """
        Removes the next candidate node from the structure and returns it.
        :return: ai_algorithms.search.graph_search.Node
        """
        raise NotImplementedError
