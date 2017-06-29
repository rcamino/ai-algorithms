class State(object):

    def agents(self):
        """
        Returns available agents.
        :return: list of ai_algorithms.environment.agent.Agent
        """
        raise NotImplementedError

    def next_state(self, agent, action):
        """
        Creates a new state from this one after the agent performs an action.
        :param agent: must implement ai_algorithms.environment.agent.Agent
        :param action: action from the agent possible actions in this state; can be a string or any other object
        :return: ai_algorithms.environment.state.State
        """
        raise NotImplementedError
