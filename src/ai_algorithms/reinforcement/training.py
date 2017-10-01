from ai_algorithms.environment.episode import Episode, run_episode


def run_training(environment, learning_agents, episodes):
    """
    Runs several episodes where a collection of learning agents can learn about their action outcomes.
    :param environment: must implement ai_algorithms.environment.models.Environment
    :param learning_agents: set of agents; they must implement ai_algorithms.reinforcement.learning_agent.LearningAgent
    :param episodes: number of episodes to be run
    """
    for _ in xrange(episodes):
        run_episode(environment, TrainingEpisode(learning_agents))


class TrainingEpisode(Episode):

    def __init__(self, learning_agents):
        """
        :param learning_agents: set of agents; they must ai_algorithms.reinforcement.learning_agent.LearningAgent
        """
        self.learning_agents = learning_agents

    def state_changed(self, environment, state, agent, action, new_state):
        """
        When a state changes, if the agent that took the action is a learning agent,
        it tells him to learn from the outcome.
        :param environment: must implement ai_algorithms.environment.models.Environment
        :param state: current state; must be a hashable object
        :param agent: must implement ai_algorithms.environment.agent.Agent
        :param action: action that the agent performed in the state; must be a hashable object
        :param new_state: state obtained after the agent performed the action in the state; must be a hashable object
        """
        if agent in self.learning_agents:
            reward = environment.reward(state, action, new_state)
            agent.learn(environment, state, agent, action, new_state, reward)
