import numpy as np
import torch
from utils.epsilon_decay_strategies import ExponentialEpsilonDecay, LinearEpsilonDecay

class GreedyStrategy():
    def __init__(self,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.device = device

    def select_action(self, model, state):
        state_tensor = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            q_values = model(state_tensor)
            return q_values.argmax().item()

class EpsilonGreedyStrategy():
    def __init__(self, 
                 max_epsilon,
                 min_epsilon,
                 decay_rate,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.epsilon_decay = LinearEpsilonDecay(max_epsilon, min_epsilon, decay_rate)
        self.device = device
        self.epsilon = max_epsilon

    def select_action(self, model, state, current_step):
        state_tensor = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            q_values = model(state_tensor)

        self.epsilon = self.epsilon_decay.get_epsilon(current_step)
        if np.random.rand() > self.epsilon:
            return q_values.argmax().item()
        return np.random.randint(len(q_values))

    def get_epsilon(self):
        return self.epsilon

if __name__ == "__main__":
    from model import DummyNetwork
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DummyNetwork().to(device)

    state = torch.rand(4)
    current_step = 10

    greedy_strategy = GreedyStrategy()
    greedy_action = greedy_strategy.select_action(model, state)
    print("Greedy Action:", greedy_action)

    max_epsilon = 1.0
    min_epsilon = 0.1
    decay_rate = 0.01
    epsilon_greedy_strategy = EpsilonGreedyStrategy(max_epsilon, min_epsilon, decay_rate)
    epsilon_greedy_action = epsilon_greedy_strategy.select_action(model, state, current_step)
    print("Epsilon-Greedy Action:", epsilon_greedy_action)
