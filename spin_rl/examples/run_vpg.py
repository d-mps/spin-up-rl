import os
print(os.path.abspath(os.getcwd()))

import argparse
import gym
from spin_rl.algorithms.vpg import VanillaPolicyGradient


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v0')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=100)
    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parser()    

    # Define environment
    gym_env = gym.make(args.env)
    
    policy = VanillaPolicyGradient(
        env=gym_env,
        obs_dim=gym_env.observation_space.shape[0],
        action_dim=gym_env.action_space.n  # Discrete action space
    )

    policy.train(
        num_epochs=10,
        learn_rate=args.lr,
        epoch_batch_size=args.batch_size
    )
