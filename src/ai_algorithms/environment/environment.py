class Environment(object):

    def agents(self):
        """
        Returns available agents.
        :return: list of ai_algorithms.environment.agent.Agent
        """
        raise NotImplementedError

    def react(self, agent, action):
        """
        Creates a new environment from this one after the agent performs an action.
        :param agent: must implement ai_algorithms.environment.agent.Agent
        :param action: action from the agent possible actions in this environment; can be a string or any other object
        :return: ai_algorithms.environment.environment.Environment
        """
        raise NotImplementedError

    def next_agent(self):
        """
        Returns the next agent that has to chose an action.
        :return: ai_algorithms.environment.agent.Agent or None if the simulation has to end
        """
        raise NotImplementedError
