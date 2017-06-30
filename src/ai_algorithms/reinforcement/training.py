from ai_algorithms.environment.episode import Episode, run_episode


def run_training(environment, learning_agents, episodes):
    for _ in xrange(episodes):
        run_episode(environment, TrainingEpisode(learning_agents))


class TrainingEpisode(Episode):

    def __init__(self, learning_agents):
        self.learning_agents = learning_agents

    def state_changed(self, environment, state, agent, action, new_state):
        if agent in self.learning_agents:
            reward = environment.reward(state, action, new_state)
            agent.learn(environment, state, agent, action, new_state, reward)
