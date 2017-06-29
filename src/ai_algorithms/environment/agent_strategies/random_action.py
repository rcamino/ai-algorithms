import random

from ai_algorithms.environment.agent_strategies.agent_strategy import AgentStrategy


class RandomAction(AgentStrategy):

    def __init__(self, random_state=None):
        """
        :param random_state: random.RandomState; if None, default random state will be used
        """
        if random_state is None:
            random_state = random.Random()
        self.random_state = random_state

    def next_action(self, environment, agent):
        """
        Takes the next action for the agent at random from the possible actions in the environment.
        :param environment: must implement ai_algorithms.environment.environment.Environment
        :param agent: must implement ai_algorithms.environment.agent.Agent
        :return: chosen action; can be a string or any other object
        """
        return self.random_state.choice(agent.actions(environment))
