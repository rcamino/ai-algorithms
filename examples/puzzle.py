import random
import time

from ai_algorithms.model import UniformCost, Deterministic, Model
from ai_algorithms.search.a_star import a_star_search
from ai_algorithms.search.bfs import bfs_search
from ai_algorithms.search.dfs import dfs_search
from ai_algorithms.search.ucs import ucs_search


ACTIONS = {
    "left": (0, -1),
    "right": (0, 1),
    "top": (-1, 0),
    "down": (1, 0),
}


def create_solved_board(board_size):
    k = 1
    board = []
    for i in xrange(board_size):
        row = []
        for j in xrange(board_size):
            if k == board_size * board_size:
                row.append(0)
            else:
                row.append(k)
            k += 1
        board.append(tuple(row))
    return tuple(board)


def create_random_board(board_size, problem):
    board = create_solved_board(board_size)
    moves = random.randint(1, 50)
    for _ in xrange(moves):
        actions = problem.actions_from(board)
        action = random.choice(actions)
        board = problem.next_state(board, action)
    return board


def print_state(state, board_size):
    line = "-"
    for j in xrange(board_size * 2):
        line += "-"
    print line
    for i in xrange(board_size):
        row = "|"
        for j in xrange(board_size):
            if state[i][j] == 0:
                row += " "
            else:
                row += str(state[i][j])
            row += "|"
        print row
        print line


class Puzzle(Model, Deterministic, UniformCost):

    def __init__(self, board_size, initial_board=None):
        self.board_size = board_size
        if initial_board is None:
            self.initial_board = create_random_board(board_size, self)
        else:
            self.initial_board = initial_board

    def find_empty_space(self, state):
        for i in xrange(self.board_size):
            for j in xrange(self.board_size):
                if state[i][j] == 0:
                    return i, j
        raise Exception("Empty space is missing.")

    def actions_from(self, state):
        i, j = self.find_empty_space(state)
        actions = []
        for action, (action_i, action_j) in ACTIONS.items():
            if 0 <= i + action_i < self.board_size and 0 <= j + action_j < self.board_size:
                actions.append(action)
        return actions

    def start_state(self):
        return self.initial_board

    def is_goal_state(self, state):
        k = 1
        for i in xrange(self.board_size):
            for j in xrange(self.board_size):
                if state[i][j] != k:
                    return False
                k += 1
                if k == self.board_size * self.board_size:
                    return True
        return True

    def next_state(self, state, action):
        empty_i, empty_j = self.find_empty_space(state)
        action_i, action_j = ACTIONS[action]
        swap_i, swap_j = empty_i + action_i, empty_j + action_j

        new_board = []
        for i in xrange(self.board_size):
            row = []
            for j in xrange(self.board_size):
                if i == empty_i and j == empty_j:
                    row.append(state[i + action_i][j + action_j])
                elif i == swap_i and j == swap_j:
                    row.append(0)
                else:
                    row.append(state[i][j])
            new_board.append(tuple(row))
        return tuple(new_board)


def hamming_distance(problem, state):
    distance = 0
    k = 0
    for i in xrange(problem.board_size):
        for j in xrange(problem.board_size):
            tile = state[i][j]
            if tile != k and tile != 0:
                distance += 1
            k += 1
    return distance


def manhattan_distance(problem, state):
    distance = 0
    k = 0
    for i in xrange(problem.board_size):
        for j in xrange(problem.board_size):
            tile = state[i][j]
            if tile != k and tile != 0:
                distance += abs(i - tile / problem.board_size) + abs(j - tile % problem.board_size)
            k += 1
    return distance


def main():
    board_size = 3
    total_puzzles = 10

    total_time = {
        "BFS": 0,
        "DFS": 0,
        "UCS": 0,
        "A* Hamming": 0,
        "A* Manhattan": 0,
    }

    algorithms = {
        "BFS": bfs_search,
        "DFS": dfs_search,
        "UCS": ucs_search,
        "A* Hamming": lambda p: a_star_search(p, hamming_distance),
        "A* Manhattan": lambda p: a_star_search(p, manhattan_distance),
    }

    for i in xrange(1, total_puzzles + 1):
        problem = Puzzle(board_size)
        print "Solving puzzle", i, "/", total_puzzles

        for name, algorithm in algorithms.items():
            start_time = time.time()
            algorithm(problem)
            total_time[name] += time.time() - start_time

    print
    print "Sorted by total time:"
    for name, t in sorted(total_time.items(), key=lambda kv: kv[1], reverse=True):
        print name, "average", t / float(total_puzzles), "total", t


if __name__ == "__main__":
    main()
