from ai_algorithms.environment.agent import Agent
from ai_algorithms.environment.agent_strategies.agent_strategy import AgentStrategy


class LearningAgent(Agent):

    def learn(self, environment, state, agent, action, new_state, reward):
        """
        Learn about the outcome of an action taken by an agent.
        This is delegated to the strategy.
        :param environment: must implement ai_algorithms.environment.models.Environment
        :param state: current state; must be a hashable object
        :param agent: must implement ai_algorithms.environment.agent.Agent
        :param action: action that the agent performed in the state; must be a hashable object
        :param new_state: state obtained after the agent performed the action in the state; must be a hashable object
        :param reward: obtained after the agent performed the action in the state; int or float
        """
        self.strategy.learn(environment, state, agent, action, new_state, reward)


class LearningAgentStrategy(AgentStrategy):

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
        raise NotImplementedError
