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
    parser.add_argument('--lr', type=float)
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()
    print(args)
    # TRAIN
