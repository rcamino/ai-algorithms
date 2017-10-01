from ai_algorithms.environment.agent_strategies.agent_strategy import AgentStrategy


class Minimax(AgentStrategy):

    def __init__(self, depth):
        """
        :param depth: int >= 0 indicating the maximum amount of node levels to explore
        """
        self.depth = depth

    def next_action(self, environment, state, agent):
        """
        :param environment: must implement ai_algorithms.environment.models.Environment
        :param state: current state; must be a hashable object
        :param agent: must implement ai_algorithms.environment.agent.Agent
        :return: chosen action; must be a hashable object
        """
        agents = [agent] + [other_agent for other_agent in sorted(environment.agents(state)) if agent != other_agent]
        _, action = self.next_transition(environment, state, self.depth, agents, 0)
        return action

    def next_transition(self, environment, state, depth, agents, agent_index):
        if depth == 0 or len(environment.actions_from(state)) == 0:
            return environment.evaluate(state, agents[0]), None
        else:
            if agent_index == 0:
                return self.max_value_transition(environment, state, depth, agents)
            elif agent_index < len(agents):
                return self.min_value_transition(environment, state, depth, agents, agent_index)
            else:
                return self.next_transition(environment, state, depth - 1, agents, 0)

    def max_value_transition(self, environment, state, depth, agents):
        agent = agents[0]
        max_value = -float("inf")
        max_action = None
        for action in environment.actions_from(state):
            next_state = environment.react(state, agent, action)
            value, _ = self.next_transition(environment, next_state, depth, agents, 1)
            if max_action is None or value > max_value:
                max_value = value
                max_action = action
        return max_value, max_action

    def min_value_transition(self, environment, state, depth, agents, agent_index):
        agent = agents[agent_index]
        min_value = float("inf")
        min_action = None
        for action in environment.actions_from(state):
            next_state = environment.react(state, agent, action)
            value, _ = self.next_transition(environment, next_state, depth, agents, agent_index + 1)
            if min_action is None or value < min_value:
                min_value = value
                min_action = action
        return min_value, min_action
