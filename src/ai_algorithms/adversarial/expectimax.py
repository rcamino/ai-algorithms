from ai_algorithms.environment.agent_estimator import UniformActions
from ai_algorithms.environment.agent_strategies.agent_strategy import AgentStrategy


class Expectimax(AgentStrategy):

    def __init__(self, environment, depth, agent_estimator=None):
        """
        :param model: must implement ai_algorithms.environment.environment.Environment
        :param depth: int >= 0 indicating the maximum amount of node levels to explore
        :param agent_estimator: must implement ai_algorithms.environment.agent_estimator.AgentEstimator
        """
        self.environment = environment
        self.depth = depth
        if agent_estimator is None:
            self.agent_estimator = UniformActions()
        else:
            self.agent_estimator = agent_estimator

    def next_action(self, state, agent):
        """
        :param state: current state; must be a hashable object
        :param agent: must implement ai_algorithms.environment.agent.Agent
        :return: chosen action; must be a hashable object
        """
        agents = [agent] + [other_agent for other_agent in sorted(self.environment.agents()) if agent != other_agent]
        _, action = self.next_transition(state, self.depth, agents, 0)
        return action

    def next_transition(self, state, depth, agents, agent_index):
        if depth == 0 or agent_index < len(agents) and len(agents[agent_index].actions(state)) == 0:
            return self.environment.evaluate(state, agents[0]), None
        else:
            if agent_index == 0:
                return self.max_value_transition(state, depth, agents)
            elif agent_index < len(agents):
                return self.expected_value(state, depth, agents, agent_index), None
            else:
                return self.next_transition(state, depth - 1, agents, 0)

    def max_value_transition(self, state, depth, agents):
        agent = agents[0]
        max_value = -float("inf")
        max_action = None
        for action in agent.actions(state):
            next_state = self.environment.react(state, agent, action)
            value, _ = self.next_transition(next_state, depth, agents, 1)
            if max_action is None or value > max_value:
                max_value = value
                max_action = action
        return max_value, max_action

    def expected_value(self, state, depth, agents, agent_index):
        agent = agents[agent_index]
        result = 0.0
        actions = agent.actions(state)
        for action in actions:
            next_state = self.environment.react(state, agent, action)
            value, _ = self.next_transition(next_state, depth, agents, agent_index + 1)
            result += value * self.agent_estimator.agent_action_probability(state, agent, actions, action)
        return result
