import random

class Teacher:
    """ 
    A class to implement a teacher that knows the optimal playing strategy.
    Works with any N×N tic-tac-toe board (default: 4x4).

    Parameters
    ----------
    level : float 
        teacher ability level (0–1): probability of playing optimally.
    board_size : int
        size of the square board (default: 4)
    """

    def __init__(self, level=0.9, board_size=4):
        self.ability_level = level
        self.board_size = board_size

    def win(self, board, key='X'):
        """Find a winning move: N-1 in a row and 1 empty."""
        n = self.board_size

        # Check rows
        for i in range(n):
            row = board[i]
            if row.count(key) == n - 1 and row.count('-') == 1:
                return (i, row.index('-'))

        # Check columns
        for j in range(n):
            col = [board[i][j] for i in range(n)]
            if col.count(key) == n - 1 and col.count('-') == 1:
                return (col.index('-'), j)

        # Check diagonals
        diag1 = [board[i][i] for i in range(n)]
        if diag1.count(key) == n - 1 and diag1.count('-') == 1:
            idx = diag1.index('-')
            return (idx, idx)

        diag2 = [board[i][n - 1 - i] for i in range(n)]
        if diag2.count(key) == n - 1 and diag2.count('-') == 1:
            idx = diag2.index('-')
            return (idx, n - 1 - idx)

        return None

    def blockWin(self, board):
        """Block the opponent if they can win."""
        return self.win(board, key='O')

    def center(self, board):
        """Pick the center (or nearest to center for even boards)."""
        n = self.board_size
        centers = []
        if n % 2 == 1:
            centers = [(n // 2, n // 2)]
        else:
            centers = [(n//2 - 1, n//2 - 1), (n//2 - 1, n//2),
                       (n//2, n//2 - 1), (n//2, n//2)]
        for (i, j) in centers:
            if board[i][j] == '-':
                return (i, j)
        return None

    def corner(self, board):
        """Pick a corner if available."""
        n = self.board_size
        corners = [(0, 0), (0, n-1), (n-1, 0), (n-1, n-1)]
        for (i, j) in corners:
            if board[i][j] == '-':
                return (i, j)
        return None

    def sideEmpty(self, board):
        """Pick a non-corner side cell if available."""
        n = self.board_size
        positions = []
        for i in range(n):
            for j in range(n):
                # edge but not corner
                if (i in [0, n-1] or j in [0, n-1]) and (i, j) not in [(0,0),(0,n-1),(n-1,0),(n-1,n-1)]:
                    if board[i][j] == '-':
                        positions.append((i, j))
        if positions:
            return random.choice(positions)
        return None

    def randomMove(self, board):
        """Pick any random empty cell."""
        n = self.board_size
        possibles = [(i, j) for i in range(n) for j in range(n) if board[i][j] == '-']
        if possibles:
            return random.choice(possibles)
        return None

    def makeMove(self, board):
        """
        Teacher goes through hierarchy of strategies:
        1. Win
        2. Block opponent's win
        3. Take center
        4. Take corner
        5. Take side
        6. Random move
        """
        if random.random() > self.ability_level:
            return self.randomMove(board)

        move = (self.win(board) or
                self.blockWin(board) or
                self.center(board) or
                self.corner(board) or
                self.sideEmpty(board) or
                self.randomMove(board))
        return move
