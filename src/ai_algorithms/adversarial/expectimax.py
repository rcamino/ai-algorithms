from ai_algorithms.environment.agent_strategies.agent_strategy import AgentStrategy


class Expectimax(AgentStrategy):

    def __init__(self, state_evaluator, agent_estimator, depth):
        self.state_evaluator = state_evaluator
        self.agent_estimator = agent_estimator
        self.depth = depth

    def next_action(self, state, agent):
        agents = [agent] + [other_agent for other_agent in sorted(state.agents()) if agent != other_agent]
        _, action = self.next_transition(state, self.depth, agents, 0)
        return action

    def next_transition(self, state, depth, agents, agent_index):
        if depth == 0 or agent_index < len(agents) and len(agents[agent_index].actions(state)) == 0:
            return self.state_evaluator.evaluate(state), None
        else:
            if agent_index == 0:
                return self.max_value_transition(state, depth, agents)
            elif agent_index < state.agent_count():
                return self.expected_value(state, depth, agents, agent_index), None
            else:
                return self.next_transition(state, depth - 1, agents, 0)

    def max_value_transition(self, state, depth, agents):
        agent = agents[0]
        max_value = -float("inf")
        max_action = None
        for action in agent.actions(state):
            next_state = state.next_state(agent, action)
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
            next_state = state.next_state(agent, action)
            value, _ = self.next_transition(next_state, depth, agents, agent_index + 1)
            result += value * self.agent_estimator.agent_action_probability(state, agent, action)
        return result
