from ai_algorithms.model import Model


class Environment(Model):

    def actions_from(self, state):
        return self.actions(state, self.next_agent(state))

    def agents(self, state):
        """
        Returns available agents in the given state.
        :param state: current state; must be a hashable object
        :return: list of ai_algorithms.environment.agent.Agent
        """
        raise NotImplementedError

    def react(self, state, agent, action):
        """
        Creates a new state from a given one after the agent performs an action.
        :param state: current state; must be a hashable object
        :param agent: must implement ai_algorithms.environment.agent.Agent
        :param action: action from the agent possible actions in the state; must be a hashable object
        :return: new state; must be a hashable object
        """
        raise NotImplementedError

    def next_agent(self, state):
        """
        Returns the next agent that has to chose an action in the given state.
        :param state: current state; must be a hashable object
        :return: ai_algorithms.environment.agent.Agent
        """
        raise NotImplementedError

    def evaluate(self, state, agent):
        """
        Gives a score for a the agent situation in the given state.
        :param state: current state; must be a hashable object
        :param agent: must implement ai_algorithms.environment.agent.Agent
        :return: float or int
        """
        raise NotImplementedError


class AgentEstimator(object):

    def agent_action_probability(self, state, agent, actions, action):
        """
        Gives an estimation for the probability of an agent taking an action from the possible actions in a state.
        :param state: current state; must be a hashable object
        :param agent: must implement ai_algorithms.environment.agent.Agent
        :param actions: possible action list
        :param action: chosen action from the list; must be a hashable object
        :return: float in [0, 1]
        """
        raise NotImplementedError


class UniformActions(AgentEstimator):

    def agent_action_probability(self, state, agent, actions, action):
        """
        Gives the same probability to every possible action of an agent in a state.
        :param state: current state; must be a hashable object
        :param agent: must implement ai_algorithms.environment.agent.Agent
        :param actions: possible action list
        :param action: chosen action from the list; must be a hashable object
        :return: float in [0, 1]
        """
        return 1.0 / len(actions)
