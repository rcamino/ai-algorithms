import random

from ai_algorithms.adversarial.mcts import MonteCarloTreeSearch
from ai_algorithms.environment.agent_strategies.random_action import RandomAction

from othello import OthelloAgent, Othello, run_timed_episodes


def main():
    episodes = 10

    random_state = random.Random()

    player_w = OthelloAgent("W", MonteCarloTreeSearch())
    player_b = OthelloAgent("B", RandomAction(random_state))

    environment = Othello(player_w, player_b)

    run_timed_episodes(environment, episodes)


if __name__ == "__main__":
    main()
