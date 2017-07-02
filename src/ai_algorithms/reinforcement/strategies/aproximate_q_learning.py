from collections import defaultdict

from ai_algorithms.reinforcement.learning_agent import LearningAgentStrategy


class ApproximateQLearning(LearningAgentStrategy):

    def __init__(self, feature_extractor, learning_rate, discount):
        """
        :param feature_extractor: must implement ai_algorithms.reinforcement.feature_extractor.FeatureExtractor
        :param learning_rate: indicates the balance between to new and old experiences; int or float in [0, 1]
        :param discount: indicates how much reward should be retained from a previous step; int or float in [0, 1]
        """
        self.feature_extractor = feature_extractor
        self.learning_rate = learning_rate
        self.discount = discount
        self.weights = defaultdict(lambda: 0.0)

    def approximate_q_value(self, environment, state, agent, action):
        """
        Calculates an approximation of the optimum reward that can be obtained from a state and action pair,
        from a linear combination of a state feature vector a weight vector.
        :param environment: must implement ai_algorithms.environment.models.Environment
        :param state: current state; must be a hashable object
        :param agent: must implement ai_algorithms.environment.agent.Agent
        :param action: action that the agent selected in the state; must be a hashable object
        :return: approximation of optimum reward value that can be obtained from the state and action pair
        """
        q_value = 0.0
        features = self.feature_extractor.features(environment, state, agent, action)
        for feature_name, feature_value in features.items():
            q_value += self.weights[feature_name] * feature_value
        return q_value

    def next_action(self, environment, state, agent):
        """
        Decides the next action for the agent to chose from the possible actions in the environment,
        based on approximations of the optimum reward that can be obtained from every state and action pair.
        :param environment: must implement ai_algorithms.environment.models.Environment
        :param state: current state; must be a hashable object
        :param agent: must implement ai_algorithms.environment.agent.Agent
        :return: chosen action; must be a hashable object
        """
        max_q_value = 0.0
        max_action = None
        for action in environment.actions_from(state):
            q_value = self.approximate_q_value(environment, state, agent, action)
            if max_action is None or q_value > max_q_value:
                max_action = action
                max_q_value = q_value
        return max_action

    def learn(self, environment, state, agent, action, new_state, reward):
        """
        Learn about the outcome of an action taken by an agent by updating state feature weights.
        :param environment: must implement ai_algorithms.environment.models.Environment
        :param state: current state; must be a hashable object
        :param agent: must implement ai_algorithms.environment.agent.Agent
        :param action: action that the agent performed in the state; must be a hashable object
        :param new_state: state obtained after the agent performed the action in the state; must be a hashable object
        :param reward: obtained after the agent performed the action in the state; int or float
        """
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
