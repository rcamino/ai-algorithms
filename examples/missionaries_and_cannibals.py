from collections import namedtuple

from ai_algorithms.model import Model, Deterministic, NoCost
from ai_algorithms.search.bfs import bfs_search


State = namedtuple("State", ["missionaries_crossed", "cannibals_crossed", "boat_crossed"])
Action = namedtuple("Action", ["move_missionaries", "move_cannibals"])

TOTAL_MISSIONARIES = 3
TOTAL_CANNIBALS = 3
BOAT_CAPACITY = 2


def state_is_valid(state):
    if 0 < state.missionaries_crossed < state.cannibals_crossed:
        return False
    if 0 < TOTAL_MISSIONARIES - state.missionaries_crossed < TOTAL_CANNIBALS - state.cannibals_crossed:
        return False
    return True


def print_action(action):
    move = []

    if action.move_missionaries > 0:
        move.append("{} missionaries".format(action.move_missionaries))

    if action.move_cannibals > 0:
        move.append("{} cannibals".format(action.move_cannibals))

    print "<- {} ->".format(" ".join(move))


def print_state(state):
    crossed = []
    not_crossed = []

    if state.missionaries_crossed > 0:
        crossed.append("{} missionaries".format(state.missionaries_crossed))

    if TOTAL_MISSIONARIES - state.missionaries_crossed > 0:
        not_crossed.append("{} missionaries".format(TOTAL_MISSIONARIES - state.missionaries_crossed))

    if state.cannibals_crossed > 0:
        crossed.append("{} cannibals".format(state.cannibals_crossed))

    if TOTAL_CANNIBALS - state.cannibals_crossed > 0:
        not_crossed.append("{} cannibals".format(TOTAL_CANNIBALS - state.cannibals_crossed))

    if state.boat_crossed:
        crossed.insert(0, "boat")
    else:
        not_crossed.append("boat")

    print "{} ___ {}".format(" ".join(not_crossed), " ".join(crossed))


class MissionariesAndCannibals(Model, Deterministic, NoCost):

    def actions_from(self, state):
        actions = []
        for move_missionaries in xrange(TOTAL_MISSIONARIES):
            for move_cannibals in xrange(TOTAL_CANNIBALS):
                if 1 <= move_missionaries + move_cannibals <= BOAT_CAPACITY:
                    action = Action(move_missionaries=move_missionaries, move_cannibals=move_cannibals)
                    next_state = self.next_state(state, action)
                    if state_is_valid(next_state):
                        actions.append(action)
        return actions

    def start_state(self):
        return State(missionaries_crossed=0, cannibals_crossed=0, boat_crossed=False)

    def is_goal_state(self, state):
        return state.missionaries_crossed == TOTAL_MISSIONARIES \
               and state.cannibals_crossed == TOTAL_CANNIBALS \
               and state.boat_crossed

    def next_state(self, state, action):
        if state.boat_crossed:
            return State(missionaries_crossed=state.missionaries_crossed - action.move_missionaries,
                         cannibals_crossed=state.cannibals_crossed - action.move_cannibals,
                         boat_crossed=False)
        else:
            return State(missionaries_crossed=state.missionaries_crossed + action.move_missionaries,
                         cannibals_crossed=state.cannibals_crossed + action.move_cannibals,
                         boat_crossed=True)


def main():
    problem = MissionariesAndCannibals()
    actions = bfs_search(problem)

    state = problem.start_state()
    print_state(state)
    for action in actions:
        print_action(action)
        state = problem.next_state(state, action)
        print_state(state)


if __name__ == "__main__":
    main()
