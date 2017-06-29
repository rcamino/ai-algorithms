class EnvironmentEvaluator(object):
    """
    Gives a score for the environment and agent pair.
    It should give a low score if the agent is losing and a high score if the agent is winning in the given environment.
    """

    def evaluate(self, environment, agent):
        """
        :param environment: must implement ai_algorithms.environment.environment.Environment
        :param agent: must implement ai_algorithms.environment.agent.Agent
        :return: float or int
        """
        raise NotImplementedError
