class Agent(object):

    def __init__(self, strategy):
        """
        :param strategy: must implement ai_algorithms.environment.agent_strategies.agent_strategy.AgentStrategy
        """
        self.strategy = strategy

    def next_action(self, state):
        """
        Decides the next action to chose from the possible actions in the state.
        This is delegated to the strategy.
        :param state: must implement ai_algorithms.environment.state.State
        :return: chosen action; can be a string or any other object
        """
        return self.strategy.next_action(state, self)

    def actions(self, state):
        """
        Possible actions for the agent in the state.
        :param state: must implement ai_algorithms.environment.state.State
        :return: list of actions; they can be strings or any other objects
        """
        raise NotImplementedError
