from abc import ABC, abstractmethod
import math

class EpsilonDecayStrategy(ABC):
    def __init__(self, max_epsilon, min_epsilon, decay_rate):
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        self.epsilon = max_epsilon

    @abstractmethod
    def get_epsilon(self, current_step):
        return NotImplemented


class LinearEpsilonDecay(EpsilonDecayStrategy):

    def get_epsilon(self, current_step):
        return max(self.min_epsilon, self.max_epsilon - self.decay_rate * current_step)


class ExponentialEpsilonDecay(EpsilonDecayStrategy):

    def get_epsilon(self, current_step):
        return max(self.min_epsilon, self.max_epsilon * math.exp(-self.decay_rate * current_step))


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def plot_epsilon(decay_strategy):
        episodes = range(1, 1001)
        epsilons = [decay_strategy.get_epsilon(step) for step in episodes]

        plt.plot(episodes, epsilons)
        plt.xlabel('Episodes')
        plt.ylabel('Epsilon')
        plt.title('Epsilon Decay')
        plt.savefig(f'epsilon_decay_{type(decay_strategy).__name__}.png')
        plt.show()

    plot_epsilon(LinearEpsilonDecay(max_epsilon=1.0, min_epsilon=0.01, decay_rate=0.001))
    plot_epsilon(ExponentialEpsilonDecay(max_epsilon=1.0, min_epsilon=0.01, decay_rate=0.005))
