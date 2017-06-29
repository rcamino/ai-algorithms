from ai_algorithms.environment.agent_strategies.agent_strategy import AgentStrategy


class FollowPolicy(AgentStrategy):

    def __init__(self, policy):
        """
        :param policy: dictionary of environment to action
        """
        self.policy = policy

    def next_action(self, state, agent):
        """
        Takes the next action for the agent according to a policy given the current state.
        :param state: current state; must be a hashable object
        :param agent: must implement ai_algorithms.environment.agent.Agent
        :return: chosen action; must be a hashable object
        """
        return self.policy[state]
