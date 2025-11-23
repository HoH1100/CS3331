import random
import numpy as np
import argparse
from utils import init_agent, save_agent, load_agent, compute_Q, compute_label, check_exp
from gomoku_game import init_game, make_move, get_reward
import os


def training(agent1, agent2, config, save_path, verb=[0, 0]):
	agents = [agent1, agent2]
	agent_exps = [[], []]
	running = [0, 0]
	for i in range(config['epoch']):
		state, available = init_game(config['width'])

		# start playing
		count = 0
		stop = False
		y_pre = np.zeros((config['width']**2,))
		X_riv = state.reshape(2 * config['width']**2,)

		# play the game
		while not stop:
			for player, agent in enumerate(agents):
				count += 1
				# predict q value size: [1, width ** 2]
				qval = agent.predict(state.reshape(1, 2 * config['width']**2))
				# epsilon greedy to select an action
				if random.random() < config['epsilon']:
					while True:
						x = np.random.randint(config['width'])
						y = np.random.randint(config['width'])
						action = (x, y)
						if available[action] == 0:
							break
				else:
					# add available to avoid taking the place that is already taken
					index = np.argmax(qval + available.reshape((1, config['width']**2)))
					action = (int(index / config['width']), index % config['width'])

				# take the action and compute the reward of it
				new_state, new_available = make_move(state, available, action, player)
				
				reward = get_reward(new_state, player, config['win_reward'], config['lose_reward'], \
										config['even_reward'], config['keepgoing_reward'])

				# compute the target output value y of the agents
				maxQ, max_furtherQ = compute_Q(agents, player, state, new_state, \
													new_available, config['width'])

				y, y_riv = compute_label(maxQ, max_furtherQ, player, action, reward, config['gamma'], config['gamma2'], \
											qval, y_pre, config['keepgoing_reward'], config['width'])

				X = state.reshape(2 * config['width']**2,)

				# update with experience reply
				X_train, y_train = check_exp(agent_exps, player, config['buffersize'], X, y, running, config['batch_size'])

				if len(X_train) != 0:
					agent.fit(X_train, y_train, batch_size=config['batch_size'], epochs=1, verbose=verb[0])

				# update the rival if necessary, i.e. game terminate
				if y_riv is not None:
					Xriv_train, yriv_train = check_exp(agent_exps, 1 - player, config['buffersize'], X_riv, y_riv, running, config['batch_size'])

					if len(X_train) != 0:
						agents[1 - player].fit(Xriv_train, yriv_train, batch_size=config['batch_size'], epochs=1, verbose=verb[1])

				X_riv, y_pre = state.reshape(2 * config['width']**2,), y
				state, available = new_state, new_available

				# check if the game terminate
				if reward[player] != config['keepgoing_reward'] or count > config['width']**2 - 2:
					stop = True
					break

		if i % 100 == 0:
			path_1 = os.path.join(save_path, "{}_{}.pkl".format(config['agent_name_1'], i))
			save_agent(agent1, path_1)
			path_2 = os.path.join(save_path, "{}_{}.pkl".format(config['agent_name_2'], i))
			save_agent(agent2, path_2)
   
			if i > 0:
				last_player_weights_1 = agent1.layers[-1].get_weights()
				agent1.layers[-1].set_weights([
					np.random.randn(*w.shape) * 0.01 for w in last_player_weights_1
				])
   
				last_player_weights_2 = agent2.layers[-1].get_weights()
				agent2.layers[-1].set_weights([
					np.random.randn(*w.shape) * 0.01 for w in last_player_weights_2
				])

		if config['epsilon'] >= 0.1:
			config['epsilon'] -= 0.7/config['epoch']

		log_msg = 'Epoch: {}, step: {}'.format(i, count)
		print(log_msg)

		# decrease epsilon (prob of random action) every epoch
		# if epsilon > eps_threshold:
		# 	epsilon -= 2 / epoch

	return agent1, agent2


def train_agents(config, new, save_path):
	"""Create new agents or load existing agents then do training"""
	if new:
		agent1 = init_agent(config['hidden_size'], config['layer_num'], config['lr'], config['width'])
		agent2 = init_agent(config['hidden_size'], config['layer_num'], config['lr'], config['width'])
	else:
		agent1 = load_agent(config['agent_name_1'])
		agent2 = load_agent(config['agent_name_2'])

	agent1, agent2 = training(agent1, agent2, config, save_path)

	path_1 = os.path.join(save_path, "{}.pkl".format(config['agent_name_1']))
	save_agent(agent1, path_1)
	path_2 = os.path.join(save_path, "{}.pkl".format(config['agent_name_2']))
	save_agent(agent2, path_2)