import random

from ai_algorithms.environment.agent_strategies.random_action import RandomAction
from ai_algorithms.environment.episode import run_episode, Episode

from othello import OthelloAgent, Othello


class OthelloVerboseEpisode(Episode):

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
        winner = environment.evaluate_winner(state)
        if winner is not None:
            print winner.name, "wins the game!"
        else:
            print "Tie!"


def main():
    random_state = random.Random()

    player_w = OthelloAgent("W", RandomAction(random_state))
    player_b = OthelloAgent("B", RandomAction(random_state))

    environment = Othello(player_w, player_b)
    episode = OthelloVerboseEpisode()

    run_episode(environment, episode)


if __name__ == "__main__":
    main()
