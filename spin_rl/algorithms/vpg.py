"""REINFORCE Algorithm

Description:
    The reinforce is an on-policy algorithm which samples.  

Paper:
    Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist
    reinforcement learning. Machine Learning, 8(3-4):229–256.
    https://people.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf
    http://incompleteideas.net/book/the-book.html

Resources:
    https://spinningup.openai.com/en/latest/algorithms/vpg.html
    https://github.com/rll/rllab/blob/master/rllab/algos/vpg.py
    https://github.com/openai/spinningup/blob/master/spinup/examples/pytorch/pg_math/1_simple_pg.py
    https://github.com/dennybritz/reinforcement-learning/blob/master/PolicyGradient/CliffWalk%20REINFORCE%20with%20Baseline%20Solution.ipynb
"""
import gym
import numpy as np
import torch


from collections import defaultdict
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

PROD = 1

class Policy(torch.nn.Module):
    """Policy network π(a|s,θ)."""

    def __init__(self, obs_dim: int, act_dim: int):
        """Instantiate policy network with environment specifications. 

        Args:
            in_dim (int): observation space dimension
            out_dim (int): number of actions possible
        """
        super(Policy, self).__init__()
        self.linear1 = torch.nn.Linear(obs_dim, 64)
        self.linear2 = torch.nn.Linear(64, act_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32)
        x = self.relu(self.linear1(x))
        logits = self.linear2(x)
        return logits
    
    def sample(self, obs):
        """Sample an acion from the output of the policy network.
        The log_prob is the loss function. Use negative logprob, because
        Pytorch optimizers use gradient descent even though it is common
        to find gradient ascent in the literature.

        Args:
            obs (np.array): Input specific to gym environment

        Returns:
            action (float): The action to take in the environment
            log_prob (float): log probability that satisfies the Reinforce loss function
        """
        # Output unnormalized values for each possible action
        logits = self.forward(torch.as_tensor(obs, dtype=torch.float32))
        
        # Note: this is equivalent to multinomial
        # [Ref] https://pytorch.org/docs/stable/distributions.html
        action_probs = Categorical(logits=logits)
        action = action_probs.sample()
        logprob = action_probs.log_prob(action)
        return action.item(), -logprob.item()


def reinforce(env, policy: object, num_episodes: int, gamma: float):
    """Basic Reinforce algorithm (without baseline)
    
    Credit to:
        https://github.com/dennybritz/reinforcement-learning/blob/master/PolicyGradient
        /CliffWalk%20REINFORCE%20with%20Baseline%20Solution.ipynb

    Args:
        env (gym.env): OpenAI Gym environment
        policy (object): Action policy
        n_episodes (int): Number of episodes 
        gamma (float): Timestep discount factor

    Returns:
        None
    """
    ep_rew = np.zeros(num_episodes)
    ep_steps = np.zeros(num_episodes, dtype=int)

    optimizer = torch.optim.Adam(policy.parameters(), lr=0.0007)

    # Step 1.
    # Collect set of trajectories by running action policy in the environment.
    # Trajectories are also called episodes or rollouts.
    # [Ref] https://spinningup.openai.com/en/latest/spinningup/rl_intro.html#trajectories
    for i_ep in range(num_episodes):
        rollout = {'state': [], 'action': [], 'reward': [], 'neglogprob': []}
        
        # Initial observation comes from starting distribution
        obs = env.reset()
        done = False  # episode status

        while not done:
            # Sample current policy network π(a|s,θ)
            action, neglogprob = policy.sample(obs)
            
            # Get feeback from environment based on the latest action decision
            next_obs, reward, done, info = env.step(action)

            # Store transition info
            rollout['state'].append(obs)
            rollout['action'].append(action)
            rollout['reward'].append(reward)
            rollout['neglogprob'].append(neglogprob)

            ep_rew[i_ep] += reward
            ep_steps[i_ep] += 1

            obs = next_obs

        # Step 2.
        # Fit a model to estimate return from a given timestep until the end.
        # This computes "reward-to-go" following the policy with discount factor.
        rollout_returns = torch.zeros(ep_steps[i_ep])
        for t in range(ep_steps[i_ep]):
           rollout_returns[t] = sum(gamma**i_step * rew for i_step, rew in enumerate(rollout['reward'][t:]))

        # Step 3.
        # Improve the policy via batch update instead of single step
        logits = policy.forward(rollout['state'])
        action_dist = Categorical(logits=logits)
        log_probs = action_dist.log_prob(torch.as_tensor(rollout['action'], dtype=torch.int32))
        loss = (-log_probs * rollout_returns).mean()
        print('epsiode=', i_ep, 'loss', round(loss.item(), 2), 'total return', sum(rollout['reward']))

        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    # args here
    environment_name = "CartPole-v1"
    algorithm_name = 'vpg'
    env = gym.make(environment_name)
    
    experiment_name = f'{environment_name}__{algorithm_name}'
    writer = SummaryWriter(f'runs/{experiment_name}')
    
    if PROD:
        import wandb
        wandb.init(project="baselines", sync_tensorboard=True, name=experiment_name, monitor_gym=True)
        writer = SummaryWriter(f'/tmp/{experiment_name}')
    
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    action_policy = Policy(obs_dim=obs_dim, act_dim=act_dim)
    reinforce(env, policy=action_policy, gamma=0.99, num_episodes=200)
    print('Done')
    
    
    # obs = env.reset()
    # for t in range(500):
    #     env.render()
    #     action, _ = action_policy.sample(obs)
    #     obs, reward, done, info = env.step(action)
    #     if done:
    #         print('episode finished after {} timesteps'.format(t+1))
    #         break

    env.close()
