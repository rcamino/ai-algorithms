from environment.strategy import AgentStrategy


class FollowPolicy(AgentStrategy):

    def __init__(self, policy):
        self.policy = policy

    def next_action(self, state, agent):
        return self.policy[state]
