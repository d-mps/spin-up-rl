"""Vanilla Policy Gradient Algorithm

[Resources]
https://spinningup.openai.com/en/latest/algorithms/vpg.html
https://github.com/rll/rllab/blob/master/rllab/algos/vpg.py
https://github.com/openai/spinningup/blob/master/spinup/examples/pytorch/pg_math/1_simple_pg.py
"""
import numpy as np
import torch


class VanillaPolicyGradient:
    
    def __init__(self, env, obs_dim, action_dim):
        
        # Set gym environment
        self.env = env
        # Initial policy gradient parameters, theta
        self.pi_theta = Policy(input_dim=obs_dim, output_dim=action_dim)
        # Initial value function parameters, phi
        # self.q_phi = 
        # Cache historical experiences for gradient update.
        self.batch_obs = list()
        # Sampled actions from current policy.
        self.batch_actions = list()
        # Weights for each logprob(a|s) is R(tau)
        self.batch_weights = list()
        self.batch_returns = list()

    def collect_trajectories(self, batch_size: int):
        """Collect set of trajectories D_k = {Tau_i} by running
        policy pi_k = pi(theta_k) in the environment.

        Returns:
            [type]: [description]
        """
        # Initial observation comes from a starting distribution
        # TODO: experiment with starting distribution
        obs = self.env.reset()
    
        # Store rewards which is the amount achieved by the action taken.
        episode_rewards = list()
        episode_done = False
    
        while True:
            if episode_done:
                # Compute return from episode rewards
                ep_return = sum(episode_rewards)
                self.batch_returns.append(ep_return)
                self.batch_weights.extend([ep_return] * len(episode_rewards))
                # Reset episode
                obs = self.env.reset()
                episode_rewards = list()

                if len(self.batch_actions) >= batch_size:
                    # End collection of experiences
                    break

            action = self.pi_theta.get_action_from_policy(obs)

            # Store observation
            self.batch_obs.append(obs.copy())

            # Response from environment given sampled action
            obs, reward, episode_done, _ = self.env.step(action)

            # Store actions and rewards observed
            self.batch_actions.append(action)
            episode_rewards.append(reward)

    def compute_advantage_estimates(self):
        pass

    def train(self, num_epochs: int, epoch_batch_size: int, learn_rate: float):
        optimizer = torch.optim.Adam(self.pi_theta.parameters(), lr=learn_rate)

        for epoch in range(num_epochs):
            self.collect_trajectories(batch_size=epoch_batch_size)

            epoch_obs = torch.as_tensor(self.batch_obs, dtype=torch.float32)
            epoch_actions = torch.as_tensor(self.batch_actions, dtype=torch.int32)
            epoch_weights = torch.as_tensor(self.batch_weights, dtype=torch.float32)

            # Compute loss
            logits = self.pi_theta.forward(epoch_obs)
            action_dist = torch.distributions.categorical.Categorical(logits=logits)
            log_probs = action_dist.log_prob(epoch_actions)
            batch_loss = -(log_probs * epoch_weights).mean()
            print('batch loss', round(batch_loss.item(), 2), 'reward', np.mean(self.batch_returns))

            batch_loss.backward()
            optimizer.step()

            # reset cache
            self.batch_returns = list()
            self.batch_obs = list()
            self.batch_actions = list()
            self.batch_weights = list()


class Policy(torch.nn.Module):
    """Basic policy network pi_theta a policy with parameters theta."""

    def __init__(self, input_dim: int, output_dim: int):
        """Instantiate policy network with environment specifications. 

        Args:
            in_dim (int): observation space dimension
            out_dim (int): number of actions possible
        """
        super(Policy, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, 200)
        self.linear2 = torch.nn.Linear(200, 50)
        self.linear3 = torch.nn.Linear(50, output_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        logits = self.linear3(x)
        return logits
    
    def sample_action(self, logits):
        action_distribution = torch.distributions.categorical.Categorical(logits=logits)
        sample = action_distribution.sample().item()
        return sample
    
    def get_action_from_policy(self, obs):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        logits = self.forward(obs_tensor)
        action = self.sample_action(logits)
        return action
