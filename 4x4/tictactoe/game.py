import random


class Game:
    """ 
    Tic Tac Toe Game class (generalized for any N×N board, default 4×4).
    Compatible with RL agent and optional teacher (AI opponent).
    """

    def __init__(self, agent, teacher=None, board_size=4):
        self.agent = agent
        self.teacher = teacher
        self.board_size = board_size
        # initialize the game board
        self.board = [['-' for _ in range(board_size)] for _ in range(board_size)]

    def playerMove(self):
        """
        Query player (or teacher) for a move and update the board accordingly.
        """
        n = self.board_size
        if self.teacher is not None:
            action = self.teacher.makeMove(self.board)
            if action:
                self.board[action[0]][action[1]] = 'X'
        else:
            printBoard(self.board)
            while True:
                move = input(f"Your move! Please select a row and column from 0–{n-1} "
                             f"in the format row,col: ")
                print('\n')
                try:
                    row, col = map(int, move.split(','))
                except ValueError:
                    print("INVALID INPUT! Please use the correct format (e.g., 1,2).")
                    continue
                if row not in range(n) or col not in range(n) or self.board[row][col] != '-':
                    print("INVALID MOVE! Choose again.")
                    continue
                self.board[row][col] = 'X'
                break

    def agentMove(self, action):
        """Update board according to agent's move."""
        self.board[action[0]][action[1]] = 'O'

    def checkForWin(self, key):
        """
        Check if the player/agent with token 'key' has won.
        Supports N×N board — requires full row/col/diag of same key.
        """
        n = self.board_size

        # check diagonals
        if all(self.board[i][i] == key for i in range(n)):
            return True
        if all(self.board[i][n - 1 - i] == key for i in range(n)):
            return True

        # check rows and columns
        for i in range(n):
            if all(self.board[i][j] == key for j in range(n)):
                return True
            if all(self.board[j][i] == key for j in range(n)):
                return True
        return False

    def checkForDraw(self):
        """Check whether the game has ended in a draw."""
        return all(elt != '-' for row in self.board for elt in row)

    def checkForEnd(self, key):
        """
        Checks if player/agent with token 'key' has ended the game.
        Returns:
            -1 → game continues
             0 → draw
             1 → key wins
        """
        if self.checkForWin(key):
            if self.teacher is None:
                printBoard(self.board)
                print(f"{'Player' if key == 'X' else 'RL agent'} wins!")
            return 1
        elif self.checkForDraw():
            if self.teacher is None:
                printBoard(self.board)
                print("It's a draw!")
            return 0
        return -1

    def playGame(self, player_first):
        """
        Begin the tic-tac-toe game loop.
        """
        if player_first:
            self.playerMove()

        prev_state = getStateKey(self.board)
        prev_action = self.agent.get_action(prev_state)

        # main game loop
        while True:
            self.agentMove(prev_action)
            check = self.checkForEnd('O')
            if check != -1:
                reward = check  # +1 for win, 0 for draw
                break

            self.playerMove()
            check = self.checkForEnd('X')
            if check != -1:
                reward = -1 * check  # -1 if lose, 0 if draw
                break
            else:
                reward = 0

            new_state = getStateKey(self.board)
            new_action = self.agent.get_action(new_state)
            self.agent.update(prev_state, new_state, prev_action, new_action, reward)
            prev_state, prev_action = new_state, new_action

        # Final update
        self.agent.update(prev_state, None, prev_action, None, reward)

    def start(self):
        """
        Determine who moves first, and start the game.
        """
        if self.teacher is not None:
            # Teacher mode: random first mover
            if random.random() < 0.5:
                self.playGame(player_first=False)
            else:
                self.playGame(player_first=True)
        else:
            while True:
                response = input("Would you like to go first? [y/n]: ").strip().lower()
                print('')
                if response in ('n', 'no'):
                    self.playGame(player_first=False)
                    break
                elif response in ('y', 'yes'):
                    self.playGame(player_first=True)
                    break
                else:
                    print("Invalid input. Please enter 'y' or 'n'.")


def printBoard(board):
    """Prints the game board to terminal."""
    n = len(board)
    print("    " + "   ".join(str(i) for i in range(n)) + "\n")
    for i, row in enumerate(board):
        print(f"{i}   " + "   ".join(row))
        print('')


def getStateKey(board):
    """Convert board 2D list to string key for hashing in Q-table."""
    return ''.join(''.join(row) for row in board)
