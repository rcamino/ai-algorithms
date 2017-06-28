class State(object):

    def agents(self):
        raise NotImplementedError

    def next_state(self, agent, action):
        raise NotImplementedError
