from ai_algorithms.environment.agent_strategies.agent_strategy import AgentStrategy


class AlphaBetaPruning(AgentStrategy):

    def __init__(self, state_evaluator, depth):
        self.state_evaluator = state_evaluator
        self.depth = depth

    def next_action(self, state, agent):
        agents = [agent] + [other_agent for other_agent in sorted(state.agents()) if agent != other_agent]
        _, action = self.next_transition(state, self.depth, agents, 0, -float("inf"), float("inf"))
        return action

    def next_transition(self, state, depth, agents, agent_index, alpha, beta):
        if depth == 0 or agent_index < len(agents) and len(agents[agent_index].actions(state)) == 0:
            return self.state_evaluator.evaluate(state, agents[0]), None
        else:
            if agent_index == 0:
                return self.max_value_transition(state, depth, agents, alpha, beta)
            elif agent_index < len(agents):
                return self.min_value_transition(state, depth, agents, agent_index, alpha, beta)
            else:
                return self.next_transition(state, depth - 1, agents, 0, alpha, beta)

    def max_value_transition(self, state, depth, agents, alpha, beta):
        agent = agents[0]
        max_value = -float("inf")
        max_action = None
        for action in agent.actions(state):
            next_state = state.next_state(agent, action)
            value, _ = self.next_transition(next_state, depth, agents, 1, alpha, beta)
            if value > beta:
                return value, action
            alpha = max(alpha, value)
            if max_action is None or value > max_value:
                max_value = value
                max_action = action
        return max_value, max_action

    def min_value_transition(self, state, depth, agents, agent_index, alpha, beta):
        agent = agents[agent_index]
        min_value = float("inf")
        min_action = None
        for action in agent.actions(state):
            next_state = state.next_state(agent, action)
            value, _ = self.next_transition(next_state, depth, agents, agent_index + 1, alpha, beta)
            if value < alpha:
                return value, action
            beta = min(beta, value)
            if min_action is None or value < min_value:
                min_value = value
                min_action = action
        return min_value, min_action
