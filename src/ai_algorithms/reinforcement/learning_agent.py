from ai_algorithms.environment.agent import Agent
from ai_algorithms.environment.agent_strategies.agent_strategy import AgentStrategy


class LearningAgent(Agent):

    def learn(self, environment, state, agent, action, new_state, reward):
        self.strategy.learn(environment, state, agent, action, new_state, reward)


class LearningAgentStrategy(AgentStrategy):

    def learn(self, environment, state, agent, action, new_state, reward):
        raise NotImplementedError
