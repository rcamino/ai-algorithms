import random

from ai_algorithms.environment.agent_strategies.random_action import RandomAction
from ai_algorithms.probabilities.sampling import bernoulli_trial
from ai_algorithms.reinforcement.learning_agent import LearningAgentStrategy


class EpsilonExploration(LearningAgentStrategy):

    def __init__(self, decorated, exploration_rate, random_state=None):
        """
        :param decorated: must implement ai_algorithms.reinforcement.learning_agent.LearningAgent
        :param exploration_rate: defines the probability to take a random action; float in [0, 1]
        :param random_state: random.RandomState; if None, default random state will be used
        """
        self.decorated = decorated
        self.exploration_rate = exploration_rate
        if random_state is None:
            random_state = random.Random()
        self.random_state = random_state
        self.random_action = RandomAction(random_state)

    def next_action(self, environment, state, agent):
        """
        With probability equal to the exploration rate, decides to take a random action.
        Otherwise, delegates the decision to the decorated strategy.
        :param environment: must implement ai_algorithms.environment.models.Environment
        :param state: current state; must be a hashable object
        :param agent: must implement ai_algorithms.environment.agent.Agent
        :return: chosen action; must be a hashable object
        """
        if bernoulli_trial(self.exploration_rate, self.random_state):
            return self.random_action.next_action(environment, state, agent)
        else:
            return self.decorated.next_action(environment, state, agent)

    def learn(self, environment, state, agent, action, new_state, reward):
        """
        Delegates the learning to the decorated strategy.
        :param environment: must implement ai_algorithms.environment.models.Environment
        :param state: current state; must be a hashable object
        :param agent: must implement ai_algorithms.environment.agent.Agent
        :param action: action that the agent performed in the state; must be a hashable object
        :param new_state: state obtained after the agent performed the action in the state; must be a hashable object
        :param reward: obtained after the agent performed the action in the state; int or float
        """
        self.decorated.learn(environment, state, agent, action, new_state, reward)
