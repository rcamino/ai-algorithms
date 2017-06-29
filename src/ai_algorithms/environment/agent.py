class Agent(object):

    def __init__(self, strategy):
        """
        :param strategy: must implement ai_algorithms.environment.agent_strategies.agent_strategy.AgentStrategy
        """
        self.strategy = strategy

    def next_action(self, state):
        """
        Decides the next action to chose from the possible actions in the given state.
        This is delegated to the strategy.
        :param state: current state; must be a hashable object
        :return: chosen action; must be a hashable object
        """
        return self.strategy.next_action(state, self)

    def actions(self, state):
        """
        Possible actions for the agent in the given state.
        :param state: current state; must be a hashable object
        :return: list of actions; they can be strings or any other objects
        """
        raise NotImplementedError
