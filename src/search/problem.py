class SearchProblem(object):

    def start_state(self):
        raise NotImplementedError

    def is_goal_state(self, state):
        raise NotImplementedError

    def transitions(self, state):
        raise NotImplementedError
