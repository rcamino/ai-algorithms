import time

from collections import namedtuple

from ai_algorithms.environment.episode import run_episode
from ai_algorithms.environment.models import Environment, UniformActions
from ai_algorithms.model import RewardAware
from ai_algorithms.reinforcement.learning_agent import LearningAgent


State = namedtuple("State", ["board", "player", "winner", "moves"])


class TicTacToe(Environment, UniformActions, RewardAware):
    LENGTH = 3

    def __init__(self, player_1, player_2, board_size):
        self.player_1 = player_1
        self.player_2 = player_2
        self.board_size = board_size

    def agents(self, state):
        return [self.player_1, self.player_2]

    def react(self, state, agent, action):
        player = agent.name
        if player != state.player:
            raise Exception("Invalid agent turn.")

        new_board = []
        for i in xrange(self.board_size):
            row = []
            for j in xrange(self.board_size):
                if (i, j) == action:
                    row.append(player)
                else:
                    row.append(state.board[i][j])
            new_board.append(tuple(row))
        new_board = tuple(new_board)

        if self.check_winning_move(new_board, player, action):
            return State(board=new_board, player=None, winner=player, moves=state.moves + 1)
        else:
            if agent == self.player_1:
                return State(board=new_board, player=self.player_2.name, winner=None, moves=state.moves + 1)
            else:
                return State(board=new_board, player=self.player_1.name, winner=None, moves=state.moves + 1)

    def next_agent(self, state):
        if state.winner is None:
            for agent in self.agents(state):
                if agent.name == state.player:
                    return agent
        return None

    def evaluate(self, state, agent):
        if state.winner is None:
            return 0
        elif state.winner == agent.name:
            return 1
        else:
            return -1

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

    def check_winning_move(self, board, player, action):
        i, j = action

        radius = self.LENGTH - 1

        row_count = 0
        column_count = 0
        diagonal_count = 0
        anti_diagonal_count = 0
        for steps in xrange(-radius, radius + 1):
            if 0 <= i + steps < self.board_size:
                if board[i + steps][j] == player:
                    column_count += 1
                    if column_count >= self.LENGTH:
                        return True
                else:
                    column_count = 0

                if 0 <= j + steps < self.board_size:
                    if board[i + steps][j + steps] == player:
                        diagonal_count += 1
                        if diagonal_count >= self.LENGTH:
                            return True
                    else:
                        diagonal_count = 0

                if 0 <= j - steps < self.board_size:
                    if board[i + steps][j - steps] == player:
                        anti_diagonal_count += 1
                        if anti_diagonal_count >= self.LENGTH:
                            return True
                    else:
                        anti_diagonal_count = 0
            if 0 <= j + steps < self.board_size:
                if board[i][j + steps] == player:
                    row_count += 1
                    if row_count >= self.LENGTH:
                        return True
                else:
                    row_count = 0

        return False

    def start_state(self):
        board = []
        for i in xrange(self.board_size):
            row = []
            for j in xrange(self.board_size):
                row.append(None)
            board.append(tuple(row))
        return State(board=tuple(board), player=self.player_1.name, winner=None, moves=0)

    def is_goal_state(self, state):
        return (state.winner is not None) or state.moves == self.board_size * self.board_size

    def actions_from(self, state):
        result = []
        if not self.is_goal_state(state):
            for i in xrange(self.board_size):
                for j in xrange(self.board_size):
                    if state.board[i][j] is None:
                        result.append((i, j))
        return result

    def reward(self, state_from, action, state_to):
        if state_to.player == self.player_1.name:
            agent = self.player_1
        else:
            agent = self.player_2
        return self.evaluate(state_to, agent) - self.evaluate(state_from, agent)


class TicTacToeAgent(LearningAgent):

    def __init__(self, name, *args, **kwargs):
        super(TicTacToeAgent, self).__init__(*args, **kwargs)
        self.name = name

    def __repr__(self):
        return str(self.name)[0]


def run_timed_episodes(environment, episodes):
    total_time = 0.0
    wins = {environment.player_1.name: 0, environment.player_2.name: 0}

    for _ in xrange(episodes):
        start_time = time.time()

        final_state = run_episode(environment)

        if final_state.winner is not None:
            wins[final_state.winner] += 1

        elapsed_time = time.time() - start_time
        total_time += elapsed_time

        print "Game finished in {:.2f} (seconds)".format(elapsed_time)
        if final_state.winner is not None:
            print final_state.winner, "wins the game!"
        else:
            print "Tie!"

    print
    print "Total time: {:.2f} (seconds)".format(total_time)
    print "Average time: {:.2f} (seconds)".format(total_time / episodes)
    print
    print "Scoreboard"
    for k, v in wins.items():
        print k, v, "/", episodes
