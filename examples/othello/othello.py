import time

from collections import namedtuple

from ai_algorithms.environment.episode import run_episode
from ai_algorithms.environment.models import Environment, UniformActions
from ai_algorithms.model import RewardAware
from ai_algorithms.reinforcement.learning_agent import LearningAgent


State = namedtuple("State", ["board", "player"])


class Othello(Environment, UniformActions, RewardAware):

    board_size = 8
    directions = -1, 0, 1

    def __init__(self, player_1, player_2):
        self.player_1 = player_1
        self.player_2 = player_2

    def print_state(self, state):
        board = state.board
        line = "-"
        for j in xrange(self.board_size * 2):
            line += "-"
        print line
        for i in xrange(self.board_size):
            row = "|"
            for j in xrange(self.board_size):
                if board[i][j] is None:
                    row += " "
                else:
                    row += board[i][j]
                row += "|"
            print row
            print line

    def is_inside_board(self, i, j):
        return 0 <= i < self.board_size and 0 <= j < self.board_size

    def flips_after(self, state, action):
        if state.player == self.player_1.name:
            player = self.player_1.name
            opponent = self.player_2.name
        else:
            player = self.player_2.name
            opponent = self.player_1.name

        board = state.board

        action_i, action_j = action

        flips = []

        for direction_i in self.directions:
            for direction_j in self.directions:
                if direction_i == 0 and direction_j == 0:
                    continue

                direction_flips = []

                i = action_i + direction_i
                j = action_j + direction_j

                while self.is_inside_board(i, j) and board[i][j] == opponent:
                    direction_flips.append((i, j))
                    i += direction_i
                    j += direction_j

                if len(direction_flips) > 0 and self.is_inside_board(i, j) and board[i][j] == player:
                    flips.extend(direction_flips)

        return flips

    def actions_from(self, state):
        actions = []
        for i in xrange(self.board_size):
            for j in xrange(self.board_size):
                action = i, j
                if state.board[i][j] is None and len(self.flips_after(state, action)) > 0:
                    actions.append(action)

        return actions

    def is_goal_state(self, state):
        return len(self.actions_from(state)) == 0

    def start_state(self):
        board = []
        for i in xrange(self.board_size):
            row = []
            for j in xrange(self.board_size):
                row.append(None)
            board.append(row)

        board[3][3] = self.player_1.name
        board[3][4] = self.player_2.name
        board[4][3] = self.player_2.name
        board[4][4] = self.player_1.name

        board = map(tuple, board)
        return State(board=tuple(board), player=self.player_1.name)

    def evaluate(self, state, agent):
        disks = 0
        for i in xrange(self.board_size):
            for j in xrange(self.board_size):
                if state.board[i][j] == agent.name:
                    disks += 1
        return disks

    def react(self, state, agent, action):
        player = agent.name
        if player != state.player:
            raise Exception("Invalid agent turn.")

        action_i, action_j = action
        if state.board[action_i][action_j] is not None:
            raise Exception("Invalid action: non empty space.")

        flips = self.flips_after(state, action)

        if len(flips) == 0:
            raise Exception("Invalid action: no flips.")

        board = []
        for i in xrange(self.board_size):
            row = []
            for j in xrange(self.board_size):
                row.append(state.board[i][j])
            board.append(row)

        board[action_i][action_j] = player

        for i, j in flips:
            board[i][j] = player

        board = tuple(map(tuple, board))

        if agent == self.player_1:
            return State(board=board, player=self.player_2.name)
        else:
            return State(board=board, player=self.player_1.name)

    def reward(self, state_from, action, state_to):
        if state_to.player == self.player_1.name:
            agent = self.player_1
        else:
            agent = self.player_2
        return self.evaluate(state_to, agent) - self.evaluate(state_from, agent)

    def next_agent(self, state):
        for agent in self.agents(state):
            if agent.name == state.player:
                return agent

    def agents(self, state):
        return [self.player_1, self.player_2]

    def evaluate_winner(self, state):
        disks_by_agent = {
            self.player_1: self.evaluate(state, self.player_1),
            self.player_2: self.evaluate(state, self.player_2),
        }
        if disks_by_agent[self.player_1] > disks_by_agent[self.player_2]:
            return self.player_1
        elif disks_by_agent[self.player_1] < disks_by_agent[self.player_2]:
            return self.player_2
        else:
            return None


class OthelloAgent(LearningAgent):

    def __init__(self, name, *args, **kwargs):
        super(OthelloAgent, self).__init__(*args, **kwargs)
        self.name = name

    def __repr__(self):
        return str(self.name)[0]


def run_timed_episodes(environment, episodes):
    total_time = 0.0
    wins = {"W": 0, "B": 0}

    for _ in xrange(episodes):
        start_time = time.time()

        final_state = run_episode(environment)

        winner = environment.evaluate_winner(final_state)
        if winner is not None:
            wins[winner.name] += 1

        elapsed_time = time.time() - start_time
        total_time += elapsed_time

        print "Game finished in {:.2f} (seconds)".format(elapsed_time)
        if winner is not None:
            print winner.name, "wins the game!"
        else:
            print "Tie!"

    print
    print "Total time: {:.2f} (seconds)".format(total_time)
    print "Average time: {:.2f} (seconds)".format(total_time / episodes)
    print
    print "Scoreboard"
    for k, v in wins.items():
        print k, v, "/", episodes
