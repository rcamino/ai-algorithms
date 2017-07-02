def run_episode(environment, episode=None):
    """
    Runs agent actions in the environment from the initial state until a goal state is reached.
    The episode object is used as an optional hook for every event.
    :param environment: must implement ai_algorithms.environment.models.Environment
    :param episode: optional episode to run; must implement ai_algorithms.environment.episode.Episode
    :return: last state; must be a hashable object
    """
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
        """
        Called when the episode starts.
        Does nothing by default.
        :param environment: must implement ai_algorithms.environment.models.Environment
        :param state: initial state; must be a hashable object
        """
        pass

    def agent_chosen(self, environment, state, agent):
        """
        Called when a new agent is chosen to take the next action.
        Does nothing by default.
        :param environment: must implement ai_algorithms.environment.models.Environment
        :param state: current state; must be a hashable object
        :param agent: chosen agent; must implement ai_algorithms.environment.agent.Agent
        """
        pass

    def action_chosen(self, environment, state, agent, action):
        """
        Called when the last chosen agent selects an action, but before performing it.
        Does nothing by default.
        :param environment: must implement ai_algorithms.environment.models.Environment
        :param state: current state; must be a hashable object
        :param agent: must implement ai_algorithms.environment.agent.Agent
        :param action: action that the agent selected in the state; must be a hashable object
        """
        pass

    def state_changed(self, environment, state, agent, action, new_state):
        """
        Called when the last chosen agent performs an action leading to a new state.
        Does nothing by default.
        :param environment: must implement ai_algorithms.environment.models.Environment
        :param state: current state; must be a hashable object
        :param agent: must implement ai_algorithms.environment.agent.Agent
        :param action: action that the agent performed in the state; must be a hashable object
        :param new_state: state obtained after the agent performed the action in the state; must be a hashable object
        """
        pass

    def finished(self, environment, state):
        """
        Called when the episode ends by reaching a goal state.
        Does nothing by default.
        :param environment: must implement ai_algorithms.environment.models.Environment
        :param state: last state; must be a hashable object
        """
        pass
