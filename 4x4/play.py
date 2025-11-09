import argparse
import os
import pickle
import sys

from tictactoe.agent import Qlearner, SARSAlearner
from tictactoe.teacher import Teacher
from tictactoe.game import Game


class GameLearning:
    """
    A class that holds the state of the learning process.
    Learning agents are created/loaded here, and a count is kept of the
    games that have been played.
    """

    def __init__(self, args, alpha=0.5, gamma=0.9, epsilon=0.1):
        self.board_size = args.board_size

        # --- Load or create agent ---
        if args.load:
            if not os.path.isfile(args.path):
                raise ValueError("Cannot load agent: file does not exist.")
            with open(args.path, 'rb') as f:
                agent = pickle.load(f)
        else:
            if os.path.isfile(args.path):
                print(f"An agent is already saved at {args.path}.")
                while True:
                    response = input("Are you sure you want to overwrite? [y/n]: ").strip().lower()
                    if response in ('y', 'yes'):
                        break
                    elif response in ('n', 'no'):
                        print("OK. Quitting.")
                        sys.exit(0)
                    else:
                        print("Invalid input. Please choose 'y' or 'n'.")
            if args.agent_type == "q":
                agent = Qlearner(alpha, gamma, epsilon, eps_decay=0.001, board_size=self.board_size)
            else:
                agent = SARSAlearner(alpha, gamma, epsilon, eps_decay=0.001, board_size=self.board_size)

        self.games_played = 0
        self.path = args.path
        self.agent = agent

    # ------------------------------------------------------------------
    def beginPlaying(self):
        """Loop through game iterations with a human player."""
        print(f"Welcome to {self.board_size}×{self.board_size} Tic-Tac-Toe.")
        print("You are 'X' and the computer is 'O'.")

        def play_again():
            print(f"Games played: {self.games_played}")
            while True:
                play = input("Do you want to play again? [y/n]: ").strip().lower()
                if play in ('y', 'yes'):
                    return True
                elif play in ('n', 'no'):
                    return False
                else:
                    print("Invalid input. Please choose 'y' or 'n'.")

        while True:
            game = Game(self.agent, board_size=self.board_size)
            game.start()
            self.games_played += 1
            self.agent.save(self.path)
            if not play_again():
                print("OK. Quitting.")
                break

    # ------------------------------------------------------------------
    def beginTeaching(self, episodes):
        """Loop through self-play games using a Teacher."""
        teacher = Teacher(board_size=self.board_size)
        while self.games_played < episodes:
            game = Game(self.agent, teacher=teacher, board_size=self.board_size)
            game.start()
            self.games_played += 1
            if self.games_played % 1000 == 0:
                print(f"Games played: {self.games_played}")
        self.agent.save(self.path)


# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play or train Tic-Tac-Toe (N×N version).")

    parser.add_argument('-a', "--agent_type", type=str, default="q",
                        choices=['q', 's'],
                        help="Agent type: 'q' for Q-learning, 's' for SARSA-learning.")
    parser.add_argument("-p", "--path", type=str, required=False,
                        help="Path for the agent pickle file. Defaults depend on agent type.")
    parser.add_argument("-l", "--load", action="store_true",
                        help="Load existing trained agent.")
    parser.add_argument("-t", "--teacher_episodes", default=None, type=int,
                        help="Train with teacher for N episodes.")
    parser.add_argument("-b", "--board_size", default=4, type=int,
                        help="Set the board size (default 4 for 4×4 Tic-Tac-Toe).")

    args = parser.parse_args()

    # default model save path
    if args.path is None:
        args.path = f"{args.agent_type}_agent_{args.board_size}x{args.board_size}.pkl"

    # initialize game-learning session
    gl = GameLearning(args)

    # training or playing
    if args.teacher_episodes is not None:
        gl.beginTeaching(args.teacher_episodes)
    else:
        gl.beginPlaying()
