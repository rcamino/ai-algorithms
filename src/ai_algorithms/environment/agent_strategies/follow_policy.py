from ai_algorithms.environment.agent_strategies.agent_strategy import AgentStrategy


class FollowPolicy(AgentStrategy):

    def __init__(self, policy, fallback=None):
        """
        :param policy: dictionary of environment to action
        :param fallback: optional strategy to use in case there is no policy for a state
        """
        self.policy = policy
        self.fallback = fallback

    def next_action(self, environment, state, agent):
        """
        Takes the next action for the agent according to a policy given the current state.
        If the policy does not have an action for the state, the fallback strategy is used.
        If there is no fallback strategy, an exception is raised.
        :param environment: must implement ai_algorithms.environment.models.Environment
        :param state: current state; must be a hashable object
        :param agent: must implement ai_algorithms.environment.agent.Agent
        :return: chosen action; must be a hashable object
        """
        if state not in self.policy:
            if self.fallback is None:
                raise Exception("State is not in policy.")
            else:
                return self.fallback.next_action(environment, state, agent)
        return self.policy[state]
