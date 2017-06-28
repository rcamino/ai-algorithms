import random

from environment.agent_strategies.agent_strategy import AgentStrategy


class RandomAction(AgentStrategy):

    def __init__(self, random_state=None):
        if random_state is None:
            random_state = random.getstate()
        self.random_state = random_state

    def next_action(self, state, agent):
        return self.random_state.choice(agent.actions(state))
