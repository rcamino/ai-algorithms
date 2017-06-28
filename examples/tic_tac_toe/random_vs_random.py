import random

from ai_algorithms.environment.agent_strategies.random_action import RandomAction

from tic_tac_toe import TicTacToeEvaluator, TicTacToeAgent, TicTacToe, print_board


if __name__ == "__main__":
    BOARD_SIZE = 3

    random_state = random.Random()
    evaluator = TicTacToeEvaluator()

    player_O = TicTacToeAgent("O", RandomAction(random_state))
    player_X = TicTacToeAgent("X", RandomAction(random_state))

    game = TicTacToe(player_O, player_X, BOARD_SIZE)
    while game.winner is None and len(game.available_positions()) > 0:
        action = game.current_player.next_action(game)
        print game.current_player, "plays", action
        game = game.next_state(game.current_player, action)
        print_board(game.board, BOARD_SIZE)
        print

    if game.winner is None:
        print "Tie!"
    else:
        print game.winner, "wins the game!"
