# fun_with_rl

## Setup
```
pip install -e .
```

## Quick Start

Next, you can choose the environment and algorithm to train:

```
cd CartPole-v1/ddqn
python3 ddqn_cartpole.py --mode train --algorithm dqn
```

For testing:
```
cd CartPole-v1/ddqn
python3 ddqn_cartpole.py --mode test --algorithm dqn
```
