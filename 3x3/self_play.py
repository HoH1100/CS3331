from tictactoe.agent import Qlearner, SARSAlearner
from tictactoe.game import Game
import pickle
import os

def self_play(episodes=20000, alpha=0.5, gamma=0.9, eps=0.1):
    agent1 = Qlearner(alpha, gamma, eps)
    agent2 = Qlearner(alpha, gamma, eps)

    for episode in range(1, episodes + 1):
        # reset game
        game = Game(agent1)
        game2 = Game(agent2)

        board = [['-','-','-'],['-','-','-'],['-','-','-']]
        game.board = board
        game2.board = board

        turn = 0  

        while True:
            state = ''.join([''.join(r) for r in board])
            if turn % 2 == 0:  # agent1 turn (O)
                a = agent1.get_action(state)
                board[a[0]][a[1]] = 'O'
                r = game.checkForEnd('O')
            else:              # agent2 turn (X)
                a = agent2.get_action(state)
                board[a[0]][a[1]] = 'X'
                r = game2.checkForEnd('X')

            if r != -1:
                reward1 = r
                reward2 = -r
                agent1.update(state, None, a, None, reward1)
                agent2.update(state, None, a, None, reward2)
                break

            turn += 1
            new_state = ''.join([''.join(r) for r in board])
            if turn % 2 == 1:
                a2 = agent2.get_action(new_state)
                agent1.update(state, new_state, a, a2, 0)
            else:
                a1 = agent1.get_action(new_state)
                agent2.update(state, new_state, a, a1, 0)

        # chỉ in khi đủ 1000 trận
        if episode % 1000 == 0:
            print(f"Self train: {episode}")

    # lưu file
    with open("agent1.pkl","wb") as f: pickle.dump(agent1,f)
    with open("agent2.pkl","wb") as f: pickle.dump(agent2,f)

    print("Self-play training complete!")

if __name__ == "__main__":
    self_play(20000)
