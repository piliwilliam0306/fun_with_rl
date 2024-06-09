import argparse
from agent import QLearningAgent
from env import CustomCartPoleEnv
import imageio
import matplotlib.pyplot as plt
import numpy as np
import optuna
import yaml

def plot_rewards(rewards, num_episodes):
    rewards_per_thousand_episodes = np.split(np.array(rewards), num_episodes/1000)
    avg_rewards_per_thousand_episodes = []

    for r in rewards_per_thousand_episodes:
        avg_rewards_per_thousand_episodes.append(sum(r/1000))
    
    plt.plot(range(1000, num_episodes + 1, 1000), avg_rewards_per_thousand_episodes)
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.title('Average Reward vs Episodes')
    filename = 'rewards_plot.png'
    plt.savefig(filename)

def train_agent(env, agent, episodes):
    rewards = []
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.update_q_table(state, action, reward, next_state)
            state = next_state
            total_reward += reward
        agent.update_epsilon(episode)
        rewards.append(total_reward)
        if episode % 1000 == 0:
            print(episode, total_reward)
    return rewards

def evaluate_agent(env, agent, num_episodes=100, record_video=False):
    total_rewards = 0
    if record_video:
        images = []
    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = np.argmax(agent.q_table[state])
            next_state, reward, done, _, _ = env.step(action)
            total_rewards += reward
            state = next_state
            if record_video:
                img = env.render()
                images.append(img)
    if record_video:
        imageio.mimsave("cartpole.gif", [np.array(img) for i, img in enumerate(images)], fps=15)
    return total_rewards / num_episodes

def objective(trial):
    # Search space for hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.8)
    discount_factor = trial.suggest_float('discount_factor', 0.8, 0.99)
    max_epsilon = trial.suggest_float('max_epsilon', 0.5, 1.0)
    min_epsilon = trial.suggest_float('min_epsilon', 0.01, 0.1)
    decay_rate = trial.suggest_float('decay_rate', 0.0001, 0.01)

    env = CustomCartPoleEnv(10)
    n_bins = env.n_bins
    n_actions = env.action_space.n
    agent = QLearningAgent(n_bins, n_actions, learning_rate, discount_factor, max_epsilon, min_epsilon, decay_rate)

    episodes = 3000
    train_agent(env, agent, episodes)

    return evaluate_agent(env, agent)

def load_best_params(filepath):
    with open(filepath, 'r') as f:
        return yaml.load(f, Loader=yaml.SafeLoader)

def save_best_params(params, filepath):
    with open(filepath, 'w') as f:
        yaml.dump(params, f)

def main():
    parser = argparse.ArgumentParser(description='Train or test the Q-learning agent.')
    parser.add_argument('--mode', choices=['optimize', 'train', 'test', 'record'], required=True, help='Mode: optimize, train, test or record')
    parser.add_argument('--qtable', type=str, default='best_q_table.pkl', help='Path to save/load the Q-table')
    parser.add_argument('--params', type=str, default='best_params.yaml', help='Path to save/load the best parameters')
    args = parser.parse_args()

    if args.mode == 'optimize':
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=100)
        print('Best hyperparameters: ', study.best_params)
        save_best_params(study.best_params, args.params)
    elif args.mode == 'train':
        best_params = load_best_params(args.params)
        env = CustomCartPoleEnv(10)
        n_bins = env.n_bins
        n_actions = env.action_space.n
        agent = QLearningAgent(n_bins, n_actions, best_params['learning_rate'], best_params['discount_factor'], best_params['max_epsilon'], best_params['min_epsilon'], best_params['decay_rate'])

        episodes = 150000
        rewards = train_agent(env, agent, episodes)
        plot_rewards(rewards, episodes)
        agent.save_q_table(args.qtable)
    elif args.mode == 'test':
        env = CustomCartPoleEnv(10, render_mode="human")
        n_bins = env.n_bins
        n_actions = env.action_space.n
        agent = QLearningAgent(n_bins, n_actions)
        agent.load_q_table(args.qtable)
        average_reward = evaluate_agent(env, agent, num_episodes=5)
        print('Average reward with loaded Q-table: ', average_reward)
    elif args.mode == 'record':
        env = CustomCartPoleEnv(10, render_mode="rgb_array")
        n_bins = env.n_bins
        n_actions = env.action_space.n
        agent = QLearningAgent(n_bins, n_actions)
        agent.load_q_table(args.qtable)
        average_reward = evaluate_agent(env, agent, num_episodes=1, record_video=True)


if __name__ == '__main__':
    main()