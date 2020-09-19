import argparse
import torch
import gym


# define a policy network
class PiNet(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(PiNet, self).__init__()
        self.linear1 = torch.nn.Linear(in_dim, 200)
        self.linear2 = torch.nn.Linear(200, 50)
        self.linear3 = torch.nn.Linear(50, out_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        logits = self.linear3(x)
        return logits


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v0')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()

    # Create environment for agent.
    env = gym.make(args.env)
    # Environment works for continuous state space
    obs_space = env.observation_space
    obs_dim = obs_space.shape[0]
    # Environment works for Discrete action space, eg. Discrete(2)
    num_actions = env.action_space.n

    pi_theta = PiNet(in_dim=obs_dim, out_dim=num_actions)

    def get_action_distribution(obs):
        """Compute action distribution from current policy"""
        logits = pi_theta(obs)

    # TRAIN
    for epoch in range(args.epochs):
        continue
        # logging outputs
        # log observations
        # log actions taken
        # log returns or weighting in policy gradient
        # log episode returns
        # log episode number of batches, each batch is a set of experiences


