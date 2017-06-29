class AgentStrategy(object):

    def next_action(self, state, agent):
        """
        Decides the next action for the agent to chose from the possible actions in the state.
        :param state: must implement ai_algorithms.environment.state.State
        :param agent: must implement ai_algorithms.environment.agent.Agent
        :return: chosen action; can be a string or any other object
        """
        raise NotImplementedError
