import argparse
import os
import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt


def plot_agent_reward(rewards, board_size=3, save_path=None):
    """
    Plot cumulative and smoothed reward curves for the trained agent.

    Parameters
    ----------
    rewards : list or np.ndarray
        Sequence of rewards from training.
    board_size : int
        Size of the Tic-Tac-Toe board (3, 4, etc.)
    save_path : str or None
        Optional path to save the plot image (e.g., 'reward_plot.png').
    """

    if not rewards or len(rewards) == 0:
        print("⚠️ No rewards found in agent data.")
        return

    rewards = np.array(rewards)
    cumulative = np.cumsum(rewards)

    # --- Smoothed reward (moving average) ---
    window = max(1, len(rewards) // 100)  # roughly 1% window
    smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(cumulative, color='royalblue')
    plt.title(f'Cumulative Reward ({board_size}×{board_size})')
    plt.ylabel('Total Reward')
    plt.xlabel('Episode')

    plt.subplot(1, 2, 2)
    plt.plot(smoothed, color='orange')
    plt.title(f'Smoothed Reward (window={window})')
    plt.ylabel('Avg Reward')
    plt.xlabel('Episode')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"✅ Saved reward plot to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot agent training rewards.")
    parser.add_argument("-p", "--path", type=str, required=True,
                        help="Path to the saved agent pickle file.")
    parser.add_argument("-s", "--save", type=str, required=False,
                        help="Optional path to save plot as image (e.g. 'plot.png').")
    args = parser.parse_args()

    if not os.path.isfile(args.path):
        print("❌ Cannot load agent: file does not exist.")
        sys.exit(0)

    with open(args.path, 'rb') as f:
        agent = pickle.load(f)

    # Try to detect board size from agent
    board_size = getattr(agent, 'board_size', 3)

    # Validate reward data
    if not hasattr(agent, 'rewards'):
        print("⚠️ Agent does not contain 'rewards' attribute. Cannot plot.")
        sys.exit(0)

    plot_agent_reward(agent.rewards, board_size=board_size, save_path=args.save)
