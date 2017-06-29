def run_episode(environment, episode=None):
    if episode is None:
        episode = Episode()

    state = environment.start_state()
    episode.started(environment, state)

    while not environment.is_goal_state(state):
        agent = environment.next_agent(state)
        episode.agent_chosen(environment, state, agent)

        action = agent.next_action(environment, state)
        episode.action_chosen(environment, state, agent, action)

        new_state = environment.react(state, agent, action)
        episode.state_changed(environment, state, agent, action, new_state)
        state = new_state

    episode.finished(environment, state)

    return state


class Episode(object):

    def started(self, environment, state):
        pass

    def agent_chosen(self, environment, state, agent):
        pass

    def action_chosen(self, environment, state, agent, action):
        pass

    def state_changed(self, environment, state, agent, action, new_state):
        pass

    def finished(self, environment, state):
        pass
