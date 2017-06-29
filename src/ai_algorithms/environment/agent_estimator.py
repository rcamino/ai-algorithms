class AgentEstimator(object):

    def agent_action_probability(self, state, agent, actions, action):
        raise NotImplementedError


class UniformActions(AgentEstimator):

    def agent_action_probability(self, state, agent, actions, action):
        return 1.0 / len(actions)
