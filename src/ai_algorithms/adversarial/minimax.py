from ai_algorithms.environment.agent_strategies.agent_strategy import AgentStrategy


class Minimax(AgentStrategy):

    def __init__(self, model, depth):
        """
        :param model: must implement ai_algorithms.environment.models.Environment
        :param depth: int >= 0 indicating the maximum amount of node levels to explore
        """
        self.model = model
        self.depth = depth

    def next_action(self, state, agent):
        """
        :param state: current state; must be a hashable object
        :param agent: must implement ai_algorithms.environment.agent.Agent
        :return: chosen action; must be a hashable object
        """
        agents = [agent] + [other_agent for other_agent in sorted(self.model.agents()) if agent != other_agent]
        _, action = self.next_transition(state, self.depth, agents, 0)
        return action

    def next_transition(self, state, depth, agents, agent_index):
        if depth == 0 or agent_index < len(agents) and len(agents[agent_index].actions(state)) == 0:
            return self.model.evaluate(state, agents[0]), None
        else:
            if agent_index == 0:
                return self.max_value_transition(state, depth, agents)
            elif agent_index < len(agents):
                return self.min_value_transition(state, depth, agents, agent_index)
            else:
                return self.next_transition(state, depth - 1, agents, 0)

    def max_value_transition(self, state, depth, agents):
        agent = agents[0]
        max_value = -float("inf")
        max_action = None
        for action in agent.actions(state):
            next_state = self.model.react(state, agent, action)
            value, _ = self.next_transition(next_state, depth, agents, 1)
            if max_action is None or value > max_value:
                max_value = value
                max_action = action
        return max_value, max_action

    def min_value_transition(self, state, depth, agents, agent_index):
        agent = agents[agent_index]
        min_value = float("inf")
        min_action = None
        for action in agent.actions(state):
            next_state = self.model.react(state, agent, action)
            value, _ = self.next_transition(next_state, depth, agents, agent_index + 1)
            if min_action is None or value < min_value:
                min_value = value
                min_action = action
        return min_value, min_action
