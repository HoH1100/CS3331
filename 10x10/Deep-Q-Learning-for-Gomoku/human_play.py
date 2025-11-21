from gomoku_game import init_game, make_move, get_reward, draw_grid, display_grid
from utils import load_agent
import argparse

import numpy as np


def combat_with_human(agent, width, turn, win_reward, lose_reward,
					  even_reward, keepgoing_reward):
	state, available = init_game(width)

	# decide who moves first
	if turn == 0:
		human_actor, actor = 0, 1
	else:
		human_actor, actor = 1, 0

	step = 0
	terminate = False
	print('Game start! ("end" to terminate)')

	for k in range(width**2 + 5):
		step += 1
		if turn == 0:
			# human plays
			flag = True

			while flag:
				try:
					action = raw_input(">>> Input (ex. 1, 1): ")
				except NameError:
					action = input(">>> Input (ex. 1, 1): ")

				if action == 'end':
					flag = False
					terminate = True
					print('Terminating the game...')
					break

				# determine if the input is legal
				try:
					action = tuple(map(int, action.split(',')))
					flag = available[action] != 0
				except:
					flag = True

				if flag:
					print("can't put there")

			# human terminates the game
			if terminate:
				break

			state, available = make_move(state, available, action, human_actor)
			reward = get_reward(state, human_actor, win_reward, lose_reward,
							   even_reward, keepgoing_reward)[human_actor]
			turn = 1

		else:
			# agent plays
			qval = agent.predict(state.reshape(1, 2 * width**2))
			action = (np.argmax(qval + available.reshape(1, width**2)))
			action = int(action / width), (action % width)
			print('AI taking action: %s' % (action,))

			state, available = make_move(state, available, action, actor)
			reward = get_reward(state, actor, win_reward, lose_reward,
							   even_reward, keepgoing_reward)[actor]
			turn = 0

		# show the chessboard
		display_grid(draw_grid(state))

		# check if the game proceeds
		if reward != keepgoing_reward:
			if reward > 0 and turn == 1:
				print('Human Wins!')
			else:
				print('AI Wins!')
			break

		if (step > width**2 - 9):
			print("Game lost; too many moves.")
			break


def combat(agent_name, width, turn, win_reward=500, lose_reward=-1000,
		   even_reward=-100, keepgoing_reward=-10):
	"""load the agent and play with human"""
	agent = load_agent(agent_name)
	combat_with_human(agent, width, turn, win_reward, lose_reward,
					  even_reward, keepgoing_reward)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--filename', type=str, required=True)
	parser.add_argument('--width', type=int, default=11)
	args = parser.parse_args()
	combat(args.filename, args.width, 0)
