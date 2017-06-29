class Environment(object):

    def agents(self, state):
        """
        Returns available agents in the given state.
        :param state: current state; must be a hashable object
        :return: list of ai_algorithms.environment.agent.Agent
        """
        raise NotImplementedError

    def react(self, state, agent, action):
        """
        Creates a new state from a given one after the agent performs an action.
        :param state: current state; must be a hashable object
        :param agent: must implement ai_algorithms.environment.agent.Agent
        :param action: action from the agent possible actions in the state; must be a hashable object
        :return: ai_algorithms.environment.environment.Environment
        """
        raise NotImplementedError

    def next_agent(self, state):
        """
        Returns the next agent that has to chose an action in the given state.
        :param state: current state; must be a hashable object
        :return: ai_algorithms.environment.agent.Agent or None if the simulation has to end
        """
        raise NotImplementedError

    def evaluate(self, state, agent):
        """
        Gives a score for a the agent situation in the given state.
        :param state: current state; must be a hashable object
        :param agent: must implement ai_algorithms.environment.agent.Agent
        :return: float or int
        """
        raise NotImplementedError
