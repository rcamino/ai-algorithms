class Strategy(object):

    def create_candidates(self):
        raise NotImplementedError

    def add_to_candidates(self, node, candidates):
        raise NotImplementedError
