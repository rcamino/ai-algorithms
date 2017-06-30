from collections import defaultdict

from ai_algorithms.reinforcement.learning_agent import LearningAgentStrategy


class ApproximateQLearning(LearningAgentStrategy):

    def __init__(self, feature_extractor, learning_rate, discount):
        self.feature_extractor = feature_extractor
        self.learning_rate = learning_rate
        self.discount = discount
        self.weights = defaultdict(lambda: 0.0)

    def approximate_q_value(self, environment, state, agent, action):
        q_value = 0.0
        features = self.feature_extractor.features(environment, state, agent, action)
        for feature_name, feature_value in features.items():
            q_value += self.weights[feature_name] * feature_value
        return q_value

    def next_action(self, environment, state, agent):
        max_q_value = 0.0
        max_action = None
        for action in environment.actions_from(state):
            q_value = self.approximate_q_value(environment, state, agent, action)
            if max_action is None or q_value > max_q_value:
                max_action = action
                max_q_value = q_value
        return max_action

    def learn(self, environment, state, agent, action, new_state, reward):
        max_new_q_value = 0.0
        max_new_action = None
        for new_action in environment.actions_from(new_state):
            new_q_value = self.approximate_q_value(environment, new_state, agent, new_action)
            if max_new_action is None or new_q_value > max_new_q_value:
                max_new_action = new_action
                max_new_q_value = new_q_value

        difference = (reward + self.discount * max_new_q_value) \
                     - self.approximate_q_value(environment, state, agent, action)

        features = self.feature_extractor.features(environment, state, agent, action)
        for feature_name, feature_value in features.items():
            self.weights[feature_name] += self.learning_rate * difference * feature_value
