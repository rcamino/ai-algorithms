from ai_algorithms.adversarial.alpha_beta_pruning import AlphaBetaPruning
from ai_algorithms.adversarial.minimax import Minimax

from tic_tac_toe import TicTacToeAgent, TicTacToe, run_timed_episodes


def main():
    board_size = 4
    episodes = 10

    player_o = TicTacToeAgent("O", Minimax(2))
    player_x = TicTacToeAgent("X", AlphaBetaPruning(3))

    environment = TicTacToe(player_o, player_x, board_size)

    run_timed_episodes(environment, episodes)


if __name__ == "__main__":
    main()
