import random

from ai_algorithms.environment.agent_strategies.random_action import RandomAction
from ai_algorithms.probabilities.sampling import bernoulli_trial
from ai_algorithms.reinforcement.learning_agent import LearningAgentStrategy


class EpsilonExploration(LearningAgentStrategy):

    def __init__(self, decorated, exploration_rate, random_state=None):
        self.decorated = decorated
        self.exploration_rate = exploration_rate
        if random_state is None:
            random_state = random.Random()
        self.random_state = random_state
        self.random_action = RandomAction(random_state)

    def next_action(self, environment, state, agent):
        if bernoulli_trial(self.exploration_rate, self.random_state):
            return self.random_action.next_action(environment, state, agent)
        else:
            return self.decorated.next_action(environment, state, agent)

    def learn(self, environment, state, agent, action, new_state, reward):
        self.decorated.learn(environment, state, agent, action, new_state, reward)
