from mdp import MDP


class DictionaryMDP(MDP):

    def __init__(self, states, actions, transitions):
        self.states = states
        self.actions = actions
        self.transitions = transitions

    def states(self):
        return self.states

    def actions(self):
        return self.actions

    def actions_from(self, state):
        if state in self.transitions:
            return self.transitions[state].keys()
        return []

    def states_from(self, state, action):
        if state in self.transitions and action in self.transitions[state]:
            return self.transitions[state][action].keys()
        return []

    def transition(self, state_from, action, state_to):
        if state_from in self.transitions and action in self.transitions[state_from] \
                and state_to in self.transitions[state_from][action]:
            return self.transitions[state_from][action][state_to]
        return None

    def probability(self, state_from, action, state_to):
        p, _ = self.transition(state_from, action, state_to)
        return p

    def reward(self, state_from, action, state_to):
        _, r = self.transition(state_from, action, state_to)
        return r
