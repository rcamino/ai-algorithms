class Agent(object):

    def __init__(self, strategy):
        self.strategy = strategy

    def next_action(self, state):
        return self.strategy.next_action(state, self)

    def actions(self, state):
        raise NotImplementedError
