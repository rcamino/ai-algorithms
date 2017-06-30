from collections import defaultdict

from ai_algorithms.reinforcement.learning_agent import LearningAgentStrategy


class QLearning(LearningAgentStrategy):

    def __init__(self, learning_rate, discount):
        self.learning_rate = learning_rate
        self.discount = discount
        self.q_values = defaultdict(lambda: defaultdict(lambda: 0))

    def next_action_and_q_value(self, environment, state):
        max_q_value = 0.0
        max_action = None
        for action in environment.actions_from(state):
            q_value = self.q_values[state][action]
            if max_action is None or q_value > max_q_value:
                max_action = action
                max_q_value = q_value
        return max_action, max_q_value

    def next_q_value(self, environment, state):
        _, q_value = self.next_action_and_q_value(environment, state)
        return q_value

    def next_action(self, environment, state, agent):
        action, _ = self.next_action_and_q_value(environment, state)
        return action

    def learn(self, environment, state, agent, action, new_state, reward):
        self.q_values[state][action] = \
            (1.0 - self.learning_rate) * self.q_values[state][action] \
            + self.learning_rate * (reward + self.discount * self.next_q_value(environment, new_state))
