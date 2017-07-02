class FeatureExtractor(object):

    def features(self, environment, state, agent, action):
        """
        Represents the current state of the environment, the current agent and the selected action as a vector.
        :param environment: must implement ai_algorithms.environment.models.Environment
        :param state: current state; must be a hashable object
        :param agent: must implement ai_algorithms.environment.agent.Agent
        :param action: action that the agent selected in the state; must be a hashable object
        :return: dictionary of feature values by name
        """
        raise NotImplementedError
