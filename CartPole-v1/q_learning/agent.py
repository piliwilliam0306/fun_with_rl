import numpy as np
import os
import pickle
import random

class QLearningAgent:
    def __init__(self, state_bins, n_actions, learning_rate=0, discount_factor=0, max_epsilon=0, min_epsilon=0, decay_rate=0):
        self.state_bins = state_bins
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        self.epsilon = max_epsilon
        self.init_q_table(state_bins, n_actions)

    def init_q_table(self, state_bins, n_actions):
        self.q_table = np.zeros(tuple([state_bins for _ in range(4)]) + (n_actions,))

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.n_actions) # Explore: select a random action
        else:
            pos_space, vel_space, ang_space, ang_vel_space = state
            return np.argmax(self.q_table[pos_space, vel_space, ang_space, ang_vel_space]) # Exploit: select the action with max value (greedy)

    def update_q_table(self, state, action, reward, next_state):
        next_pos_space, next_vel_space, next_ang_space, next_ang_vel_space = next_state
        td_target = reward + self.gamma * np.max(self.q_table[next_pos_space, next_vel_space, next_ang_space, next_ang_vel_space, :])
        pos_space, vel_space, ang_space, ang_vel_space = state
        td_error = td_target - self.q_table[pos_space, vel_space, ang_space, ang_vel_space, action]
        self.q_table[pos_space, vel_space, ang_space, ang_vel_space, action] += self.lr * td_error

    def update_epsilon(self, episode):
        self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay_rate * episode)

    def save_q_table(self, filename):
        if os.path.exists(filename):
            os.remove(filename)
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load_q_table(self, filename):
        with open(filename, 'rb') as f:
            self.q_table = pickle.load(f)

if __name__ == '__main__':
    agent = QLearningAgent(10, 2)
    state = [1, 1, 1, 1]
    action = 1
    print(agent.q_table[1, 1, 1, 1, 1])
