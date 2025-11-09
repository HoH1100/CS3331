from abc import ABC, abstractmethod
import os
import pickle
import collections
import numpy as np
import random


class Learner(ABC):
    """
    Parent class for Q-learning and SARSA agents.

    Parameters
    ----------
    alpha : float 
        learning rate
    gamma : float
        temporal discounting rate
    eps : float 
        probability of random action vs. greedy action
    eps_decay : float
        epsilon decay rate. Larger value = more decay
    board_size : int
        size of the tic tac toe board (default: 4)
    """
    def __init__(self, alpha, gamma, eps, eps_decay=0., board_size=4):
        # Agent parameters
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.eps_decay = eps_decay
        self.board_size = board_size

        # Possible actions correspond to the set of all (x,y) coordinate pairs
        self.actions = [(i, j) for i in range(board_size) for j in range(board_size)]

        # Initialize Q values to 0 for all state-action pairs.
        # Access value for action a, state s via Q[a][s]
        self.Q = {a: collections.defaultdict(int) for a in self.actions}

        # Keep a list of reward received at each episode
        self.rewards = []

    def get_action(self, s):
        """
        Select an action given the current game state.

        Parameters
        ----------
        s : string
            state (e.g. 'X--O--X---O---X-')
        """
        # Only consider the allowed actions (empty board spaces)
        possible_actions = [a for a in self.actions if s[a[0]*self.board_size + a[1]] == '-']
        if not possible_actions:
            return None

        if random.random() < self.eps:
            # Random choose.
            action = random.choice(possible_actions)
        else:
            # Greedy choose.
            values = np.array([self.Q[a][s] for a in possible_actions])
            # Find indices of max Q values
            ix_max = np.where(values == np.max(values))[0]
            if len(ix_max) > 1:
                ix_select = np.random.choice(ix_max, 1)[0]
            else:
                ix_select = ix_max[0]
            action = possible_actions[ix_select]

        # update epsilon (geometric decay)
        self.eps *= (1. - self.eps_decay)

        return action

    def save(self, path):
        """Pickle the agent object instance to save the agent's state."""
        if os.path.isfile(path):
            os.remove(path)
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @abstractmethod
    def update(self, s, s_, a, a_, r):
        """Abstract update function (implemented by subclasses)."""
        pass


class Qlearner(Learner):
    """
    A class to implement the Q-learning agent.
    """
    def __init__(self, alpha, gamma, eps, eps_decay=0., board_size=4):
        super().__init__(alpha, gamma, eps, eps_decay, board_size)

    def update(self, s, s_, a, a_, r):
        """
        Perform the Q-Learning update of Q values.

        Parameters
        ----------
        s : string
            previous state
        s_ : string
            new state
        a : (i,j) tuple
            previous action
        a_ : (i,j) tuple
            new action (NOT used by Q-learner)
        r : int
            reward received after executing action "a" in state "s"
        """
        # Update Q(s,a)
        if s_ is not None:
            # possible next actions
            possible_actions = [act for act in self.actions if s_[act[0]*self.board_size + act[1]] == '-']
            if possible_actions:
                Q_options = [self.Q[act][s_] for act in possible_actions]
                self.Q[a][s] += self.alpha * (r + self.gamma * max(Q_options) - self.Q[a][s])
            else:
                # no available next actions
                self.Q[a][s] += self.alpha * (r - self.Q[a][s])
        else:
            # terminal state update
            self.Q[a][s] += self.alpha * (r - self.Q[a][s])

        self.rewards.append(r)


class SARSAlearner(Learner):
    """
    A class to implement the SARSA agent.
    """
    def __init__(self, alpha, gamma, eps, eps_decay=0., board_size=4):
        super().__init__(alpha, gamma, eps, eps_decay, board_size)

    def update(self, s, s_, a, a_, r):
        """
        Perform the SARSA update of Q values.

        Parameters
        ----------
        s : string
            previous state
        s_ : string
            new state
        a : (i,j) tuple
            previous action
        a_ : (i,j) tuple
            new action
        r : int
            reward received after executing action "a" in state "s"
        """
        # Update Q(s,a)
        if s_ is not None:
            self.Q[a][s] += self.alpha * (r + self.gamma * self.Q[a_][s_] - self.Q[a][s])
        else:
            # terminal state update
            self.Q[a][s] += self.alpha * (r - self.Q[a][s])

        self.rewards.append(r)
