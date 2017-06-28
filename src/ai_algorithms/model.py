class Model(object):

    def actions_from(self, state):
        raise NotImplementedError

    def start_state(self):
        raise NotImplementedError

    def is_goal_state(self, state):
        raise NotImplementedError


class CompleteSpace(object):

    def states(self):
        raise NotImplementedError

    def actions(self):
        raise NotImplementedError


class Deterministic(object):

    def next_state(self, state, action):
        raise NotImplementedError


class Stochastic(object):

    def states_from(self, state, action):
        raise NotImplementedError

    def probability(self, state_from, action, state_to):
        raise NotImplementedError


class CostAware(object):

    def cost(self, state_from, action, state_to):
        raise NotImplementedError


class NoCost(CostAware):

    def cost(self, state_from, action, state_to):
        return 0


class RewardAware(object):

    def reward(self, state_from, action, state_to):
        raise NotImplementedError
