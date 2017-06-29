import random

from ai_algorithms.environment.agent_strategies.random_action import RandomAction
from ai_algorithms.environment.episode import run_episode, Episode

from tic_tac_toe import TicTacToeAgent, TicTacToe


class TicTacToeVerboseEpisode(Episode):

    def started(self, environment, state):
        print "game started"
        environment.print_state(state)

    def agent_chosen(self, environment, state, agent):
        print agent.name, "turn"

    def action_chosen(self, environment, state, agent, action):
        print agent.name, "plays", action

    def state_changed(self, environment, state, agent, action, new_state):
        environment.print_state(new_state)

    def finished(self, environment, state):
        print "game finished"
        if state.winner is None:
            print "Tie!"
        else:
            print state.winner, "wins the game!"


def main():
    board_size = 3

    random_state = random.Random()

    player_o = TicTacToeAgent("O", board_size, RandomAction(random_state))
    player_x = TicTacToeAgent("X", board_size, RandomAction(random_state))

    environment = TicTacToe(player_o, player_x, board_size)
    episode = TicTacToeVerboseEpisode()

    run_episode(environment, episode)


if __name__ == "__main__":
    main()
