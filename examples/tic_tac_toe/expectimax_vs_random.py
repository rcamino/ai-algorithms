import random

from ai_algorithms.adversarial.expectimax import Expectimax
from ai_algorithms.environment.agent_strategies.random_action import RandomAction

from tic_tac_toe import TicTacToeAgent, TicTacToe, run_timed_episodes


def main():
    board_size = 4
    episodes = 10

    random_state = random.Random()

    player_o = TicTacToeAgent("O", Expectimax(2))
    player_x = TicTacToeAgent("X", RandomAction(random_state))

    environment = TicTacToe(player_o, player_x, board_size)

    run_timed_episodes(environment, episodes)


if __name__ == "__main__":
    main()
