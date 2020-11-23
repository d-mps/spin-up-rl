import gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common import results_plotter
from stable_baselines3.common.callbacks import BaseCallback


"""
scalar: entropy loss
scalar: policy gradient loss
scalar: value function loss
scalar: approx KL
scalar: clip factor
scalar: loss
histogram: trainable variables
scalar: discounted rewards
scalar: learning rate
scalar: advantage
"""

import wandb
wandb.init(project="baselines")


class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
    
    def _on_step(self) -> bool:
        act_net_weights = self.locals['self'].policy.action_net.weight
        self.logger.record('policy_action_net_weight', act_net_weights)
        return True


def main():
    log_dir = "./runs/"
    env_name = 'CartPole-v1'

   # Create environment
    env = gym.make(env_name)
    env = Monitor(env, log_dir)

    # Create policy
    a2c_agent = A2C(
        'MlpPolicy',
        env,
        verbose=0,
        tensorboard_log=log_dir,
        seed=0,
        learning_rate=0.0007,
        n_steps=5,
        gamma=0.99,
        gae_lambda=1.0,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        rms_prop_eps=1e-05,
        use_rms_prop=True,
        use_sde=False,
        sde_sample_freq=-1,
        normalize_advantage=False,
        create_eval_env=False,
        policy_kwargs={"net_arch": [32]},
        device='auto',
        _init_setup_model=True)

    wandb.watch(a2c_agent)
    # Train agent
    a2c_agent.learn(total_timesteps=10000, tb_log_name="a2c", callback=TensorboardCallback());

    # Accessing and modifying model parameters
    # params = a2c_agent.get_parameters()
    # params['policy']['action_net.weight']
    # params['policy']['value_net.weight']

    # obs = env.reset()
    # for _ in range(1000):
    #     act, next_obs = model.predict(obs)
    #     obs, rew, done, info = env.step(act)
    #     env.render()
    #     if done:
    #         obs = env.reset()


if __name__ == '__main__':
    main()
