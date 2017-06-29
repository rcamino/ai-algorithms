from ai_algorithms.environment.agent_strategies.agent_strategy import AgentStrategy


class FollowPolicy(AgentStrategy):

    def __init__(self, policy):
        """
        :param policy: dictionary of environment to action
        """
        self.policy = policy

    def next_action(self, environment, agent):
        """
        Takes the next action for the agent according to a policy given the environment.
        :param environment: must implement ai_algorithms.environment.environment.Environment
        :param agent: must implement ai_algorithms.environment.agent.Agent
        :return: chosen action; can be a string or any other object
        """
        return self.policy[environment]
