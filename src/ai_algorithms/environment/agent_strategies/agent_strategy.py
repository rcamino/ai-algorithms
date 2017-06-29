class AgentStrategy(object):

    def next_action(self, environment, agent):
        """
        Decides the next action for the agent to chose from the possible actions in the environment.
        :param environment: must implement ai_algorithms.environment.environment.Environment
        :param agent: must implement ai_algorithms.environment.agent.Agent
        :return: chosen action; can be a string or any other object
        """
        raise NotImplementedError
