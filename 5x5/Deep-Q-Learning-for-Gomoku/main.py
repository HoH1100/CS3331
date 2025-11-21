import random
import numpy as np
import argparse
import sys
import os
import ast
from train import train_agents
import keras

random.seed(17)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--new', type=ast.literal_eval, default=True)
	parser.add_argument('--epoch', type=int, default=100)
	parser.add_argument('--v', type=str, default='v_1')
	args = parser.parse_args()

	paras = {'new': args.new, 'agent_name_1': 'agent_1', 'agent_name_2': 'agent_2', 'epoch': args.epoch, 
				'layer_num': 2, 'hidden_size': 256, 'gamma': 0.95, 'gamma2': 0.95, 'lr': 1e-4, 'width': 5, 'epsilon': 0.8, 
				'win_reward': 500, 'lose_reward':-500, 'even_reward':-100, 'keepgoing_reward': -10, 'buffersize': 100, 
				'batch_size': 32
		}

	save_path = "./output/{}/epoch_{}".format(args.v, args.epoch)
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	train_agents(paras, args.new, save_path)
