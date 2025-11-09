import os
import pickle
import argparse
import numpy as np
from tictactoe.agent import Qlearner, SARSAlearner
from tictactoe.game import Game
from tqdm import trange  # hiển thị tiến trình đẹp hơn


def self_play(
    episodes=20000,
    alpha=0.5,
    gamma=0.9,
    epsilon=0.1,
    agent_type="q",
    save_dir="agents",
    save_every=5000
):
    """
    Huấn luyện 2 agent chơi Tic-Tac-Toe tự động với nhau (self-play).

    Parameters
    ----------
    episodes : int
        Số ván huấn luyện.
    alpha, gamma, epsilon : float
        Tham số học cho Q-learning/SARSA.
    agent_type : str
        'q' hoặc 's' (chọn Q-learning hay SARSA).
    save_dir : str
        Thư mục lưu agent sau khi train.
    save_every : int
        Tần suất lưu file agent.
    """

    os.makedirs(save_dir, exist_ok=True)
    AgentClass = Qlearner if agent_type == "q" else SARSAlearner

    agent1 = AgentClass(alpha, gamma, epsilon)
    agent2 = AgentClass(alpha, gamma, epsilon)

    rewards_per_episode = []

    for episode in trange(1, episodes + 1, desc="Training..."):
        game = Game(agent1)
        board = [['-'] * 3 for _ in range(3)]
        game.board = board
        turn = 0
        done = False

        while not done:
            state = ''.join([''.join(r) for r in board])
            if turn % 2 == 0:  # agent1 turn (O)
                a = agent1.get_action(state)
                board[a[0]][a[1]] = 'O'
                result = game.checkForEnd('O')
            else:              # agent2 turn (X)
                a = agent2.get_action(state)
                board[a[0]][a[1]] = 'X'
                result = game.checkForEnd('X')

            if result != -1:
                # Trò chơi kết thúc: thắng = 1, thua = -1, hòa = 0
                reward1 = result
                reward2 = -result
                agent1.update(state, None, a, None, reward1)
                agent2.update(state, None, a, None, reward2)
                rewards_per_episode.append(reward1)
                done = True
            else:
                turn += 1
                new_state = ''.join([''.join(r) for r in board])
                # cập nhật tạm (reward=0 trong khi đang chơi)
                if turn % 2 == 1:
                    a2 = agent2.get_action(new_state)
                    agent1.update(state, new_state, a, a2, 0)
                else:
                    a1 = agent1.get_action(new_state)
                    agent2.update(state, new_state, a, a1, 0)

        # lưu checkpoint
        if episode % save_every == 0:
            path1 = os.path.join(save_dir, f"agent1_{agent_type}_{episode}.pkl")
            path2 = os.path.join(save_dir, f"agent2_{agent_type}_{episode}.pkl")
            with open(path1, "wb") as f1, open(path2, "wb") as f2:
                pickle.dump(agent1, f1)
                pickle.dump(agent2, f2)

    # Lưu bản cuối
    final_path1 = os.path.join(save_dir, f"agent1_{agent_type}_final.pkl")
    final_path2 = os.path.join(save_dir, f"agent2_{agent_type}_final.pkl")
    with open(final_path1, "wb") as f1, open(final_path2, "wb") as f2:
        pickle.dump(agent1, f1)
        pickle.dump(agent2, f2)

    print(f"✅ Self-play training complete! Agents saved to '{save_dir}'.")

    # Lưu thêm file reward để vẽ biểu đồ
    np.save(os.path.join(save_dir, f"rewards_{agent_type}.npy"), rewards_per_episode)
    print(f"📊 Saved reward history ({len(rewards_per_episode)} episodes).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TicTacToe agents via self-play.")
    parser.add_argument("-e", "--episodes", type=int, default=20000)
    parser.add_argument("-a", "--agent_type", choices=["q", "s"], default="q")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--save_dir", type=str, default="agents")
    args = parser.parse_args()

    self_play(
        episodes=args.episodes,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        agent_type=args.agent_type,
        save_dir=args.save_dir
    )
