from collections import namedtuple

from ai_algorithms.model import Model, Deterministic, NoCost
from ai_algorithms.search.dfs import dfs_search


State = namedtuple("State", ["fox_crossed", "goose_crossed", "beans_crossed", "man_crossed"])
Action = namedtuple("Action", ["move_fox", "move_goose", "move_beans"])

BOAT_CAPACITY = 2


def state_is_valid(state):
    if state.fox_crossed == state.goose_crossed and state.fox_crossed != state.man_crossed:
        return False
    if state.beans_crossed == state.goose_crossed and state.goose_crossed != state.man_crossed:
        return False
    return True


def print_action(action):
    move = []

    if action.move_fox:
        move.append("fox")

    if action.move_goose:
        move.append("goose")

    if action.move_beans:
        move.append("beans")

    move.append("man")

    print "<- {} ->".format(" ".join(move))


def print_state(state):
    crossed = []
    not_crossed = []

    if state.fox_crossed:
        crossed.append("fox")
    else:
        not_crossed.append("fox")

    if state.goose_crossed:
        crossed.append("goose")
    else:
        not_crossed.append("goose")

    if state.beans_crossed:
        crossed.append("beans")
    else:
        not_crossed.append("beans")

    if state.man_crossed:
        crossed.append("man")
    else:
        not_crossed.append("man")

    print "{} ___ {}".format(" ".join(not_crossed), " ".join(crossed))


class FoxGooseBeans(Model, Deterministic, NoCost):

    def actions_from(self, state):
        actions = []
        for move_fox in [True, False]:
            if move_fox and state.fox_crossed != state.man_crossed:
                continue
            for move_goose in [True, False]:
                if move_goose and state.goose_crossed != state.man_crossed:
                    continue
                for move_beans in [True, False]:
                    if move_beans and state.beans_crossed != state.man_crossed:
                        continue
                    count = 0
                    for move in [move_fox, move_goose, move_beans]:
                        if move:
                            count += 1
                    if count >= BOAT_CAPACITY:
                        continue
                    action = Action(move_fox=move_fox, move_goose=move_goose, move_beans=move_beans)
                    next_state = self.next_state(state, action)
                    if state_is_valid(next_state):
                        actions.append(action)
        return actions

    def start_state(self):
        return State(fox_crossed=False, goose_crossed=False, beans_crossed=False, man_crossed=False)

    def is_goal_state(self, state):
        return state.fox_crossed and state.goose_crossed and state.beans_crossed and state.man_crossed

    def next_state(self, state, action):
        return State(fox_crossed=not state.fox_crossed if action.move_fox else state.fox_crossed,
                     goose_crossed=not state.goose_crossed if action.move_goose else state.goose_crossed,
                     beans_crossed=not state.beans_crossed if action.move_beans else state.beans_crossed,
                     man_crossed=not state.man_crossed)


def main():
    problem = FoxGooseBeans()
    actions = dfs_search(problem)

    state = problem.start_state()
    print_state(state)
    for action in actions:
        print_action(action)
        state = problem.next_state(state, action)
        print_state(state)


if __name__ == "__main__":
    main()
