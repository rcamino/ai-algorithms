def run_episode(environment, episode):
    state = environment.start_state()
    episode.started(environment, state)

    while not environment.is_goal_state(state):
        agent = environment.next_agent(state)
        episode.agent_chosen(environment, state, agent)

        action = agent.next_action(state)
        episode.action_chosen(environment, state, agent, action)

        new_state = environment.react(state, agent, action)
        episode.state_changed(environment, state, agent, action, new_state)
        state = new_state

    episode.finished(environment, state)


class Episode(object):

    def started(self, environment, state):
        raise NotImplementedError

    def agent_chosen(self, environment, state, agent):
        raise NotImplementedError

    def action_chosen(self, environment, state, agent, action):
        raise NotImplementedError

    def state_changed(self, environment, state, agent, action, new_state):
        raise NotImplementedError

    def finished(self, environment, state):
        raise NotImplementedError
