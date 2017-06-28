import random
import time

from ai_algorithms.adversarial.minimax import Minimax
from ai_algorithms.environment.agent_strategies.random_action import RandomAction

from tic_tac_toe import TicTacToeEvaluator, TicTacToeAgent, TicTacToe


if __name__ == "__main__":
    BOARD_SIZE = 4
    TOTAL_GAMES = 20

    random_state = random.Random()
    evaluator = TicTacToeEvaluator()

    player_O = TicTacToeAgent("O", Minimax(evaluator, 1))
    player_X = TicTacToeAgent("X", RandomAction(random_state))

    wins = {player_O: 0, player_X: 0}
    total_time = 0.0

    for _ in xrange(TOTAL_GAMES):
        start_time = time.time()

        game = TicTacToe(player_O, player_X, BOARD_SIZE)
        while game.winner is None and len(game.available_positions()) > 0:
            action = game.current_player.next_action(game)
            game = game.next_state(game.current_player, action)

        if game.winner is None:
            print "Tie!"
        else:
            wins[game.winner] += 1
            print game.winner, "wins the game!"

        total_time += time.time() - start_time

    print
    print "Total time (seconds):", total_time
    print "Average time (seconds):", total_time / TOTAL_GAMES
    print
    print "Scoreboard"
    for k, v in wins.items():
        print k, v, "/", TOTAL_GAMES
