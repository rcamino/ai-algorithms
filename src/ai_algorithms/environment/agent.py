class Agent(object):

    def __init__(self, strategy):
        """
        :param strategy: must implement ai_algorithms.environment.agent_strategies.agent_strategy.AgentStrategy
        """
        self.strategy = strategy

    def next_action(self, environment):
        """
        Decides the next action to chose from the possible actions in the environment.
        This is delegated to the strategy.
        :param environment: must implement ai_algorithms.environment.environment.Environment
        :return: chosen action; can be a string or any other object
        """
        return self.strategy.next_action(environment, self)

    def actions(self, environment):
        """
        Possible actions for the agent in the environment.
        :param environment: must implement ai_algorithms.environment.environment.Environment
        :return: list of actions; they can be strings or any other objects
        """
        raise NotImplementedError
