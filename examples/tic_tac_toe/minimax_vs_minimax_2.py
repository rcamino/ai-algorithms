from ai_algorithms.adversarial.minimax import Minimax

from tic_tac_toe import TicTacToeAgent, TicTacToe, run_timed_episodes


def main():
    board_size = 4
    episodes = 10

    player_o = TicTacToeAgent("O", board_size, Minimax(1))
    player_x = TicTacToeAgent("X", board_size, Minimax(2))

    environment = TicTacToe(player_o, player_x, board_size)

    run_timed_episodes(environment, episodes)


if __name__ == "__main__":
    main()
