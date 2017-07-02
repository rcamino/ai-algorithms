from collections import defaultdict

from ai_algorithms.reinforcement.learning_agent import LearningAgentStrategy


class QLearning(LearningAgentStrategy):

    def __init__(self, learning_rate, discount):
        """
        :param learning_rate: indicates the balance between to new and old experiences; int or float in [0, 1]
        :param discount: indicates how much reward should be retained from a previous step; int or float in [0, 1]
        """
        self.learning_rate = learning_rate
        self.discount = discount
        self.q_values = defaultdict(lambda: defaultdict(lambda: 0.0))

    def next_action_and_q_value(self, environment, state):
        """
        Calculates the optimum reward that can be obtained from a state and the action that must be chosen.
        :param environment: must implement ai_algorithms.environment.models.Environment
        :param state: current state; must be a hashable object
        :return:
        """
        max_q_value = 0.0
        max_action = None
        for action in environment.actions_from(state):
            q_value = self.q_values[state][action]
            if max_action is None or q_value > max_q_value:
                max_action = action
                max_q_value = q_value
        return max_action, max_q_value

    def next_q_value(self, environment, state):
        """
        Calculates the optimum reward that can be obtained from a state.
        :param environment: must implement ai_algorithms.environment.models.Environment
        :param state: current state; must be a hashable object
        :return:
        """
        _, q_value = self.next_action_and_q_value(environment, state)
        return q_value

    def next_action(self, environment, state, agent):
        """
        Decides the next action for the agent to chose from the possible actions in the environment,
        based on the optimum reward that can be obtained from every state and action pair.
        :param environment: must implement ai_algorithms.environment.models.Environment
        :param state: current state; must be a hashable object
        :param agent: must implement ai_algorithms.environment.agent.Agent
        :return: chosen action; must be a hashable object
        """
        action, _ = self.next_action_and_q_value(environment, state)
        return action

    def learn(self, environment, state, agent, action, new_state, reward):
        """
        Learn about the outcome of an action taken by an agent.
        :param environment: must implement ai_algorithms.environment.models.Environment
        :param state: current state; must be a hashable object
        :param agent: must implement ai_algorithms.environment.agent.Agent
        :param action: action that the agent performed in the state; must be a hashable object
        :param new_state: state obtained after the agent performed the action in the state; must be a hashable object
        :param reward: obtained after the agent performed the action in the state; int or float
        """
        self.q_values[state][action] = \
            (1.0 - self.learning_rate) * self.q_values[state][action] \
            + self.learning_rate * (reward + self.discount * self.next_q_value(environment, new_state))
