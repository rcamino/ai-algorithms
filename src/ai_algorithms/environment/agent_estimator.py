class AgentEstimator(object):

    def agent_action_probability(self, environment, agent, actions, action):
        """
        Gives an estimation for the probability of an agent taking an action given the environment.
        :param environment: must implement ai_algorithms.environment.environment.Environment
        :param agent: must implement ai_algorithms.environment.agent.Agent
        :param actions: possible action list
        :param action: chosen action from the list; can be a string or any other object
        :return: float in [0, 1]
        """
        raise NotImplementedError


class UniformActions(AgentEstimator):

    def agent_action_probability(self, environment, agent, actions, action):
        """
        Gives the same probability to every possible action of an agent given the environment.
        :param environment: must implement ai_algorithms.environment.environment.Environment
        :param agent: must implement ai_algorithms.environment.agent.Agent
        :param actions: possible action list
        :param action: chosen action from the list; can be a string or any other object
        :return: float in [0, 1]
        """
        return 1.0 / len(actions)
