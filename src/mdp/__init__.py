class MDP(object):

    def states(self):
        raise NotImplementedError

    def actions(self):
        raise NotImplementedError

    def actions_from(self, state):
        raise NotImplementedError

    def states_from(self, state, action):
        raise NotImplementedError

    def probability(self, state_from, action, state_to):
        raise NotImplementedError

    def reward(self, state_from, action, state_to):
        raise NotImplementedError
