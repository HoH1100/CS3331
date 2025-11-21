from gomoku_game import init_game, make_move, get_reward, draw_grid, display_grid
from utils import load_agent
import argparse
import numpy as np


def agent_play(agent1_name, agent2_name, width, win_reward=500,
			   lose_reward=-1000, even_reward=-100,
			   keepgoing_reward=-10):
	"""Load two agents and let them play against each other"""
	agent1 = load_agent(agent1_name)
	agent2 = load_agent(agent2_name)
	play_game(agent1, agent2, width, win_reward, lose_reward,
			  even_reward, keepgoing_reward)


def play_game(agent1, agent2, width, win_reward, lose_reward,
			  even_reward, keepgoing_reward):
	"""agents will take the move with the highest Q value"""
	state, available = init_game(width)
	agents = [agent1, agent2]

	# while game still in progress
	stop = False
	step = 0

	for k in range(width**2 + 5):

		for actor in range(2):
			step += 1
			qval = agents[actor].predict(state.reshape(1, 2 * width**2))

			# policy: choose the move with the max Q value
			action = (np.argmax(qval + available.reshape(1, width**2)))
			action = int(action / width), (action % width)
			print('Move #: %s; Actor %s, taking action: %s' %
				  (step, actor, action))

			state, available = make_move(state, available, action, actor)
			display_grid(draw_grid(state))
			reward = get_reward(state, actor, win_reward, lose_reward,
							   even_reward, keepgoing_reward)

			if reward[actor] != -10:
				print("Reward: %s" % (reward,))
				stop = True
				break
		input()
		if (step > 350):
			print("Game lost; too many moves.")
			break

		if stop:
			break	


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--filename1', type=str, required=True)
	parser.add_argument('--filename2', type=str, required=True)
	parser.add_argument('--width', type=int, default=11)
	args = parser.parse_args()
	agent_play(args.filename1, args.filename2, args.width)
