import gymnasium as gym
import numpy as np

class CustomCartPoleEnv(gym.Env):
    def __init__(self, n_bins=10, render_mode=None):
        super(CustomCartPoleEnv, self).__init__()
        self.env = gym.make('CartPole-v1', render_mode=render_mode)
        self.n_bins = n_bins
        self.init_bins(n_bins)

    def init_bins(self, n_bins):
        """
        using terminal condition from:
        https://gymnasium.farama.org/environments/classic_control/cart_pole/#observation-space
        """
        self.bins = [
            np.linspace(-2.4, 2.4, n_bins),       # Cart Position
            np.linspace(-4, 4, n_bins),           # Cart Velocity
            np.linspace(-0.2095, 0.2095, n_bins), # Pole Angle
            np.linspace(-4, 4, n_bins)            # Pole Velocity At Tip
        ]

    def _discretize_state(self, states):
        state_index = []
        for state, state_bin in zip(states, self.bins):
            state_index.append(np.digitize(state, state_bin) - 1)
        return tuple(state_index)

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        discretized_state = self._discretize_state(state)
        return discretized_state, reward, terminated, truncated, info

    def reset(self):
        state, info = self.env.reset()
        discretized_state = self._discretize_state(state)
        return discretized_state, info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed)

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space
    
    @property
    def number_of_bins(self):
        return self.n_bins

if __name__ == '__main__':
    env = CustomCartPoleEnv()
    state, _ = env.reset()
    print(state)
