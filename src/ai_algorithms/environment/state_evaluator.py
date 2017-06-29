class StateEvaluator(object):
    """
    Gives a score for a state and agent pair.
    It should give a low score if the agent is losing and a high score if the agent is winning in the given state.
    """

    def evaluate(self, state, agent):
        """
        :param state: must implement ai_algorithms.environment.state.State
        :param agent: must implement ai_algorithms.environment.agent.Agent
        :return: float or int
        """
        raise NotImplementedError
