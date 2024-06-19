import argparse
import gymnasium as gym
import imageio
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from utils.model import QNetwork
from utils.replay_buffer import ReplayBuffer
from utils.action_select_strategies import EpsilonGreedyStrategy, GreedyStrategy
from utils.param_update_strategies import soft_update


def train_dqn(episodes=2000, max_t=500, gamma=0.99, tau=0.005, lr=1e-4, update_every=2, 
              buffer_size=int(1e4), batch_size=128, epsilon_start=1.0, epsilon_min=0.01, 
              epsilon_decay=0.0005, algorithm='ddqn'):

    env = gym.make('CartPole-v1')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    qnetwork_local = QNetwork(state_size, action_size).to(device)
    qnetwork_target = QNetwork(state_size, action_size).to(device)
    optimizer = optim.Adam(qnetwork_local.parameters(), lr=lr)

    replay_buffer = ReplayBuffer(state_size, action_size, buffer_size, device)

    action_strategy = EpsilonGreedyStrategy(epsilon_start, epsilon_min, epsilon_decay, device)

    writer = SummaryWriter(f'runs/{algorithm}')
    scores = []

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        truncated = False
        for t in range(max_t):
            action = action_strategy.select_action(qnetwork_local, state, episode)
            next_state, reward, done, truncated, _ = env.step(action)
            replay_buffer.push(state, action, next_state, reward, done)

            state = next_state
            total_reward += reward
            if done or truncated:
                break

            if replay_buffer.size > batch_size and t % update_every == 0:
                experiences = replay_buffer.sample(batch_size)
                states, actions, next_states, rewards, dones = experiences

                # Get expected Q values corresponding to the chosen actions for each state from local model
                q_values = qnetwork_local(states).gather(1, actions)

                if algorithm == 'dqn':
                    # Get max predicted Q values from target model
                    q_targets_next = qnetwork_target(next_states).detach().max(1).values.unsqueeze(1)
                elif algorithm == 'ddqn':
                    # Get the action that maximizes Q-values from the local model for the next states
                    next_actions = qnetwork_local(next_states).detach().argmax(1).unsqueeze(1)
                    # Get the Q-values for those actions from the target model
                    q_targets_next = qnetwork_target(next_states).detach().gather(1, next_actions)

                # Compute Q targets for current states
                q_targets = rewards + (gamma * q_targets_next * (1 - dones))

                # Compute loss
                loss = F.mse_loss(q_values, q_targets)

                # Minimize the loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update target network
                soft_update(qnetwork_local, qnetwork_target, tau)

        scores.append(total_reward)
        writer.add_scalar('Total Reward per Episode', total_reward, episode)
        writer.add_scalar('Average Reward (last 10 episodes)', np.mean(scores[-10:]), episode)
        writer.add_scalar('Epsilon', action_strategy.get_epsilon(), episode)

        if episode % 10 == 0:
            print(f"Episode {episode}\tAverage Score: {np.mean(scores[-10:])}\tEpsilon: {action_strategy.get_epsilon()}")

    writer.close()
    torch.save(qnetwork_local.state_dict(), f'{algorithm}_cartpole.pth')

def test_dqn(algorithm='ddqn', record_video=False):
    if record_video:
        env = gym.make('CartPole-v1', render_mode='rgb_array')
        images = []
    else:
        env = gym.make('CartPole-v1', render_mode='human')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    qnetwork_local = QNetwork(state_size, action_size).to(device)
    qnetwork_local.load_state_dict(torch.load(f'{algorithm}_cartpole.pth'))
    qnetwork_local.eval()

    action_strategy = GreedyStrategy(device)

    state, _ = env.reset()
    total_reward = 0
    done = False
    truncated = False
    while (not done and not truncated):
        action = action_strategy.select_action(qnetwork_local, state)
        state, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        if record_video:
            img = env.render()
            images.append(img)
    if record_video:
        imageio.mimsave(f"{algorithm}_cartpole.gif", [np.array(img) for img in (images)])
    print(f"Total Reward: {total_reward}")

def main():
    parser = argparse.ArgumentParser(description='Train or test a DQN model on CartPole-v1')
    parser.add_argument('--mode', choices=['train', 'test'], default='train', help='Mode to run: train or test')
    parser.add_argument('--algorithm', choices=['dqn', 'ddqn'], default='ddqn', help='Algorithm to use: dqn or ddqn')
    args = parser.parse_args()

    if args.mode == 'train':
        train_dqn(algorithm=args.algorithm)
    elif args.mode == 'test':
        test_dqn(algorithm=args.algorithm, record_video=True)

if __name__ == "__main__":
    main()