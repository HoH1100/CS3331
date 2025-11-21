import numpy as np 
import copy


def init_game(width):
	state = np.zeros((width, width, 2))
	available = np.zeros((width, width))
	return state, available


# specify the actor and the location of the new stone
def make_move(state, available, action, player):
	state_ret = copy.deepcopy(state)
	available_ret = copy.deepcopy(available)
	state_ret[action][player] = player+1
	available_ret[action] = float("-inf")
	return state_ret, available_ret


# check if the game winning criteria is met
def win_game(sub_state, player):
	for i in range(sub_state.shape[0]):
		for j in range(sub_state.shape[1]):
			if j+4 < sub_state.shape[1]:
				horizontal = sub_state[i][j: j+5]
				if (horizontal == (player+1)).all():
					return True

			if i+4 < sub_state.shape[0]:
				vertical = [sub_state[i+k, j] for k in range(5)]
				if (np.array(vertical) == (player+1)).all():
					return True
			
			if j+4 < sub_state.shape[1] and i+4 < sub_state.shape[0]:
				diagonal = [sub_state[(i+k, j+k)] for k in range(5)]
				if (np.array(diagonal) == (player+1)).all():
					return True
	return False


# check if the chessboard is full
def full_grid(state):
	return not ((state[:, :, 0] + state[:, :, 1]) == 0).any()


# calculate the reward given to whom just moved
def get_reward(state, whose_turn, win_reward=500, lose_reward=-1000,
					even_reward=-100, keepgoing_reward=-10):
	reward = [0, 0]
	if win_game(state[:, :, whose_turn], whose_turn):
		reward[whose_turn] = win_reward
		reward[1 - whose_turn] = lose_reward
	elif full_grid(state):
		reward = [even_reward, even_reward]
	else:
		reward[whose_turn] = keepgoing_reward

	return reward


def draw_grid(state):
	"""visualize the chessboard"""
	grid = np.zeros(state.shape[:2], dtype='<U2')
	grid[:] = ' '

	for i in range(state.shape[0]):
		for j in range(state.shape[1]):

			if (state[(i, j)] > 0).any():

				if (state[(i, j)] == 1).all():
					raise

				elif state[(i, j)][0] == 1:
					grid[(i, j)] = 'O'

				else:
					grid[(i, j)] = 'X'

	return grid


def display_grid(grid):
	"""print out the chessboard"""
	wid = grid.shape[0]
	show_num = 9 if wid > 9 else wid

	# chessboard
	line = '\n' + '- + ' * (wid - 1) + '- {}\n'
	line = line.join([' | '.join(grid[i]) for i in range(wid)])

	# mark the number of its lines
	bottom = ('\n' + '  {} ' * show_num)
	bottom = bottom.format(*[i+1 for i in range(show_num)])

	if show_num == 9:
		part = (' {} '*(wid - show_num))
		part = part.format(*[i+1 for i in range(show_num, wid)])
		bottom += part

	print(line.format(*[i+1 for i in range(wid)]) + bottom)