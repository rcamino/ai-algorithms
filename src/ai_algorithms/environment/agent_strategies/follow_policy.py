from ai_algorithms.environment.agent_strategies.agent_strategy import AgentStrategy


class FollowPolicy(AgentStrategy):

    def __init__(self, policy):
        """
        :param policy: dictionary of state to action
        """
        self.policy = policy

    def next_action(self, state, agent):
        """
        Takes the next action for the agent according to a policy given the state.
        :param state: must implement ai_algorithms.environment.state.State
        :param agent: must implement ai_algorithms.environment.agent.Agent
        :return: chosen action; can be a string or any other object
        """
        return self.policy[state]
