# Learning about deep reinforcement learning with OpenAI's Spinning Up


**[NOTE]** Currently this README serves as a notepad, but will be edited to a proper description of the repo. 


Follow instructions on the 
[spinningup install page](https://spinningup.openai.com/en/latest/user/installation.html)
to get started.

MuJoCo was not installed for the experiments in this repo.

Make sure to check your installation by their running install test.

```
python -m spinup.run ppo --hid "[32,32]" --env LunarLander-v2 --exp_name installtest --gamma 0.999
```

## Environments

### Classics
- CartPole-v0
- CartPole-v1
