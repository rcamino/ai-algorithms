from ai_algorithms.environment.environment import Environment
from ai_algorithms.environment.agent import Agent
from ai_algorithms.environment.state_evaluator import StateEvaluator


def create_empty_board(board_size):
    board = []
    for i in xrange(board_size):
        row = []
        for j in xrange(board_size):
            row.append(None)
        board.append(row)
    return board


def print_board(board, board_size):
    line = "-"
    for j in xrange(board_size * 2):
        line += "-"
    print line
    for i in xrange(board_size):
        row = "|"
        for j in xrange(board_size):
            if board[i][j] is None:
                row += " "
            else:
                row += board[i][j].name
            row += "|"
        print row
        print line


def check_winning_move(board, board_size, player, move, length=3):
    i, j = move
    radius = length - 1

    row_count = 0
    column_count = 0
    diagonal_count = 0
    anti_diagonal_count = 0
    for steps in xrange(-radius, radius + 1):
        if 0 <= i + steps < board_size:
            if board[i + steps][j] == player:
                column_count += 1
                if column_count >= length:
                    return True
            else:
                column_count = 0

            if 0 <= j + steps < board_size:
                if board[i + steps][j + steps] == player:
                    diagonal_count += 1
                    if diagonal_count >= length:
                        return True
                else:
                    diagonal_count = 0

            if 0 <= j - steps < board_size:
                if board[i + steps][j - steps] == player:
                    anti_diagonal_count += 1
                    if anti_diagonal_count >= length:
                        return True
                else:
                    anti_diagonal_count = 0
        if 0 <= j + steps < board_size:
            if board[i][j + steps] == player:
                row_count += 1
                if row_count >= length:
                    return True
            else:
                row_count = 0

    return False


class TicTacToe(Environment):

    def __init__(self, player_1, player_2, board_size, board=None, current_player=None, winner=None):
        self.player_1 = player_1
        self.player_2 = player_2
        self.board_size = board_size
        self.board = board
        if board is None:
            self.board = create_empty_board(board_size)
        else:
            self.board = board
        if current_player is None:
            self.current_player = player_1
        else:
            self.current_player = current_player
        self.winner = winner

    def available_positions(self):
        result = []
        if self.winner is None:
            for i in xrange(self.board_size):
                for j in xrange(self.board_size):
                    if self.board[i][j] is None:
                        result.append((i, j))
        return result

    def agents(self):
        return [self.player_1, self.player_2]

    def react(self, agent, action):
        if agent != self.current_player:
            raise Exception("Invalid agent turn.")

        new_board = []
        for i in xrange(self.board_size):
            row = []
            for j in xrange(self.board_size):
                if (i, j) == action:
                    row.append(agent)
                else:
                    row.append(self.board[i][j])
            new_board.append(row)

        if check_winning_move(new_board, self.board_size, agent, action):
            return TicTacToe(self.player_1, self.player_2, self.board_size, new_board, None, agent)
        else:
            if agent == self.player_1:
                next_player = self.player_2
            else:
                next_player = self.player_1

            return TicTacToe(self.player_1, self.player_2, self.board_size, new_board, next_player)


class TicTacToeAgent(Agent):

    def __init__(self, name, *args, **kwargs):
        super(TicTacToeAgent, self).__init__(*args, **kwargs)
        self.name = name

    def __repr__(self):
        return str(self.name)[0]

    def actions(self, state):
        return state.available_positions()


class TicTacToeEvaluator(StateEvaluator):

    def evaluate(self, state, agent):
        if state.winner is None:
            return 0
        elif state.winner == agent:
            return 1
        else:
            return -1
