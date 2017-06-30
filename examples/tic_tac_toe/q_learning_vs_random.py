import random

from ai_algorithms.environment.agent_strategies.follow_policy import FollowPolicy
from ai_algorithms.environment.agent_strategies.random_action import RandomAction
from ai_algorithms.mdp.q_value_iteration import policy_from_q_values
from ai_algorithms.reinforcement.strategies.epsilon_exploration import EpsilonExploration
from ai_algorithms.reinforcement.strategies.q_learning import QLearning
from ai_algorithms.reinforcement.training import run_training

from tic_tac_toe import TicTacToeAgent, TicTacToe, run_timed_episodes


def main():
    board_size = 3
    exploration = 0.5
    learning_rate = 5.0
    discount = 5.0
    training_episodes = 10000
    episodes = 10

    random_state = random.Random()

    player_x = TicTacToeAgent("X", RandomAction(random_state))

    q_learning = QLearning(learning_rate, discount)
    training_player_o = TicTacToeAgent("O", EpsilonExploration(q_learning, exploration))
    training_environment = TicTacToe(training_player_o, player_x, board_size)

    run_training(training_environment, [training_player_o], training_episodes)

    policy = policy_from_q_values(q_learning.q_values)
    evaluation_player_o = TicTacToeAgent("O", FollowPolicy(policy, RandomAction(random_state)))
    evaluation_environment = TicTacToe(evaluation_player_o, player_x, board_size)

    run_timed_episodes(evaluation_environment, episodes)


if __name__ == "__main__":
    main()
