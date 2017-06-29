class AgentStrategy(object):

    def next_action(self, environment, state, agent):
        """
        Decides the next action for the agent to chose from the possible actions in the environment.
        :param environment: must implement ai_algorithms.environment.models.Environment
        :param state: current state; must be a hashable object
        :param agent: must implement ai_algorithms.environment.agent.Agent
        :return: chosen action; must be a hashable object
        """
        raise NotImplementedError
