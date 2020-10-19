# Vanilla Policy Gradient (VPG)

VPG is an on-policy algorithm, which means it samples actions 
stochastically from a current policy and later updates its policy from a
batch of experiences collected over certain duration.

You can run the example script provided `run_vpg.py` with the command
below in your terminal:
```
python -m spin_rl.examples.run_vpg
``` 

The policy is defined as a basic feed-forward neural network, but can be expanded to more
complex architectures. Actions are sampled from a Categorical distribution able to compute
the log-likelihood.
