class Strategy(object):
    """
    Interface for graph search problems.
    A strategy must specify how to create the candidate node structure and how to add candidate nodes to it.
    """

    def create_candidates(self):
        """
        Create candidate structure.
        :return: ai_algorithms.search.candidates.Candidates
        """
        raise NotImplementedError

    def add_to_candidates(self, node, candidates):
        """
        Add a candidate node to the candidate structure.
        :param node: ai_algorithms.search.graph_search.Node
        :param candidates: ai_algorithms.search.candidates.Candidates
        """
        raise NotImplementedError
