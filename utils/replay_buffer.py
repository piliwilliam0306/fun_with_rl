import numpy as np
import torch


class ReplayBuffer(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 capacity=int(1e6),
                 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.state = np.zeros((capacity, state_dim))
        self.action = np.zeros((capacity, action_dim))
        self.next_state = np.zeros((capacity, state_dim))
        self.reward = np.zeros((capacity, 1))
        self.done = np.zeros((capacity, 1))
        self.device = device

    def push(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.done[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[indices]).to(self.device),
            torch.LongTensor(self.action[indices]).to(self.device),
            torch.FloatTensor(self.next_state[indices]).to(self.device),
            torch.FloatTensor(self.reward[indices]).to(self.device),
            torch.FloatTensor(self.done[indices]).to(self.device)
        )

    def is_full(self):
        return self.size == self.capacity


if __name__ == "__main__":
    state_dim = 4
    action_dim = 2

    buffer = ReplayBuffer(state_dim, action_dim, capacity=5)

    while not buffer.is_full():
        state = np.random.rand(state_dim)
        action = np.random.randint(action_dim)
        next_state = np.random.rand(state_dim)
        reward = np.random.rand(1)
        done = np.random.randint(2)
        buffer.push(state, action, next_state, reward, done)

    batch_size = 3
    state, action, next_state, reward, done = buffer.sample(batch_size)

    print("States:", state)
    print("-" * 50)
    print("Actions:", action)
    print("-" * 50)
    print("Next States:", next_state)
    print("-" * 50)
    print("Rewards:", reward)
    print("-" * 50)
    print("Dones:", done)
    print("-" * 50)
    print("Buffer full:", buffer.is_full())
