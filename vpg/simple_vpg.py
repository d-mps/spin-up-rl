import argparse
import torch
import gym

from torch.distributions.categorical import Categorical


# define a policy network
class PiNet(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        """

        :param in_dim: int, dimension of an observation coming from the environment
        :param out_dim: int, number possible actions an agent can take
        """
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
    optimizer = torch.optim.Adam(pi_theta.parameters(), lr=args.lr)

    # TRAIN
    for epoch in range(args.epochs):
        print(' --- ')
        print('epoch: ', epoch)
        print()
        # logging outputs
        batch_obs = []
        batch_actions = []
        episode_rewards = []
        episode_weights = []
        # log returns or weighting in policy gradient
        # log episode returns
        # log episode number of batches, each batch is a set of experiences

        # First observation comes from a starting distribution TODO: which distribution
        # Observation is a numpy object representing the environment.
        obs = env.reset()

        # Collect experiences by acting in the environment with current policy.
        # Episode is a batch of experiences
        collect_experiences = True
        #while collect_experiences:
        for i in range(2):
            print('--')
            print('experience', i)
            # save observation
            batch_obs.append(obs.copy())

            # Convert to tensor float32 from numpy to play well with net
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32)

            # Get action from current policy
            logits = pi_theta.forward(obs_tensor)
            action_distribution = Categorical(logits=logits)
            action = action_distribution.sample().item()

            # Get response from environment given sampled action
            # Reward is the amount achieved by the previous action. Goal is increase total reward.
            # Done is a boolean. An indicator that your episode has ended.
            # Info used for debugging.
            obs, reward, episode_done, info = env.step(action)
            print('obs', obs, 'reward', reward, 'done', episode_done, 'info', info)

            batch_actions.append(action)
            episode_rewards.append(reward)

            if episode_done:
                # Total results from episode
                episode_return = sum(episode_rewards)
                # Multiply return to each observation, logprob(a|s) is R(tau)
                episode_weights.append((episode_return, len(episode_rewards)))
                # Reset episode variables
                obs = env.reset()
                episode_done = False
                episode_rewards = []

        # Perform a gradient update on the current policy
        # Compute loss for batch of observations seen
        seen_observations = torch.as_tensor(batch_obs, dtype=torch.float32)
        actions_made = torch.as_tensor(batch_actions, dtype=torch.int32)
        weights = [ep_ret for ep_ret, ep_len in episode_weights for i in range(ep_len)]
        weights = torch.as_tensor(weights, dtype=torch.float32)

        logits = pi_theta.forward(seen_observations)
        action_distributions = Categorical(logits=logits)
        log_probs = action_distributions.log_prob(actions_made)
        batch_loss = -(log_probs * weights).mean()  # TODO
        print('batch loss', batch_loss)

        batch_loss.backward()
        optimizer.step()

