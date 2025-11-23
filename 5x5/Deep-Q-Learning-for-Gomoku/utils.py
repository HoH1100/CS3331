# from keras.models import Sequential, model_from_json
# from keras.layers.core import Dense, Dropout, Activation
# from keras.optimizers import SGD
# from keras.layers.advanced_activations import LeakyReLU
# import keras

from tensorflow.keras.models import Sequential, model_from_json  
from tensorflow.keras.layers import Dense, Dropout, Activation, LeakyReLU  
from tensorflow.keras.optimizers import SGD  
import tensorflow.keras as keras

import numpy as np
import pickle
import random

from gomoku_game import make_move


def init_agent(hidden_size, layers, lr=1e-3, width=20, alpha=0.1, moment=0.9, loss='mse'):
	model = Sequential()
	# model.add(Dense(2 * width**2, init='lecun_uniform', input_shape=(2 * width**2,)))
	model.add(Dense(2 * width**2, kernel_initializer='lecun_uniform', input_shape=(2 * width**2,)))
	model.add(LeakyReLU(alpha=alpha))

	for i in range(layers):
		# model.add(Dense(hidden_size, init='lecun_uniform'))
		model.add(Dense(hidden_size, kernel_initializer='lecun_uniform'))
		model.add(LeakyReLU(alpha=alpha))
		model.add(Dropout(0.2))

	# linear output layer to generate real-valued outputs
	# model.add(Dense(width**2, init='lecun_uniform'))
	model.add(Dense(width**2, kernel_initializer='lecun_uniform'))	
	model.add(Activation('linear'))

	# opt = SGD(lr=lr, momentum=moment, decay=1e-18, clipnorm=1.)
	opt = SGD(learning_rate=lr, momentum=moment, clipnorm=1.)
	model.compile(loss=loss, optimizer=opt)

	return model


# save the agent network's parameters and architecture
def save_agent(agent, filename):
	json_model = agent.to_json()
	weights = agent.get_weights()
	with open(filename, 'wb') as fout:
		pickle.dump([json_model, weights], fout, pickle.HIGHEST_PROTOCOL)


# load the agent network's parameters and architecture
def load_agent(filename, lr=1e-3, moment=0.9):
	with open(filename, 'rb') as fin:
		json_model, weights = pickle.load(fin)

	agent = model_from_json(json_model)
	agent.set_weights(weights)
	# opt = SGD(lr=lr, momentum=moment, decay=1e-18, nesterov=False)
	opt = SGD(learning_rate=lr, momentum=moment, nesterov=False)
	agent.compile(loss='mse', optimizer=opt)

	return agent


"""
	Compute Q values of each possible move.
	1st Q - to suppress the opponent; 2nd Q - max self Q in the next turn
"""
def compute_Q(agents, player, pre_state, new_state, new_available, width=19):
	# suppress rival: take the move minimizing rival's max possible Q values
	rival_Q = agents[1 - player].predict(new_state.reshape(1, 2 * width**2))
	newQ = new_available.reshape((1, width**2)) - rival_Q
	maxQ = np.max(newQ)

	# rival's reaction: assume would choose the move with max Q values
	rival_action = np.argmax(rival_Q + new_available.reshape((1, width**2)))
	rival_action = int(rival_action / width), (rival_action % width)
	further_state, further_avai = make_move(new_state, new_available,
										   rival_action, 1 - player)


	# the agent's decision: the move with max "further" Q values
	further_Q = agents[player].predict(further_state.reshape(1, 2 * width**2))
	max_furtherQ = np.max(further_Q + further_avai.reshape((1, width**2)))

	return maxQ, max_furtherQ


"""
	Compute the target output y for two agents' the deep Q-network
	The policy: balance between minimizing rival's Q (gamma)
				and maximize self's max Q in the next turn (gamma2)
"""
def compute_label(maxQ, max_furtherQ, player, action, reward,
				  gamma, gamma2, qval, y_pre, keepgoing_reward, width):
	# the game proceeds - use the policy the compute y of the move
	if reward[player] == keepgoing_reward:
		update = reward[player] + (gamma * maxQ) + (gamma2 * max_furtherQ)

		update = np.clip(update, -1000, 1000)	
  
		y_riv = None
	# the game terminates y and y_riv are just the rewards itself
	else:
		y_riv = np.array(y_pre)
		y_riv[action[0] * width + action[1]] = reward[1 - player]
		update = reward[player]

	# calculate the label of the move
	y = np.array(qval)
	y[0][action[0] * width + action[1]] = update
	y = y.reshape(width**2,)

	return y, y_riv


"""
	Use experience replay (like minibatch updating)
	to avoid catastrophic forgetting
"""
def check_exp(agent_exps, player, buffersize, X, y, running, batchsize):
	X_train = []
	y_train = []

	# store the experience if the length doesn't reach the threshold
	# i.e. the memory isn't full
	if len(agent_exps[player]) < buffersize:
		agent_exps[player].append((X, y))

	# the memory is full
	# sampling a subset of the stored experience to update the agent
	else:
		# replace the old experience with the newest one
		if running[player] < (buffersize - 1):
			running[player] += 1
		else:
			running[player] = 0

		agent_exps[player][running[player]] = (X, y)

		# randomly sample experience from memory to replay
		minibatch = random.sample(agent_exps[player], batchsize)

		for memX, memY in minibatch:
			X_train.append(memX)
			y_train.append(memY)

		X_train, y_train = np.array(X_train), np.array(y_train)

	return X_train, y_train