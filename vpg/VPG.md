# Vanilla Policy Gradient

1. define your environment eg. cartpole
2. define your observation dims and num of actions
based on the environment
policy network
3. define your policy network eg. mlp, output logits
4. define a function to compute action distribution eg. categorical
5. make a action selection function sampled from policy network (output actions)

how to train policy
batch can be consider an episode or rollout
for one epoch
log:
observations, actions, R(tau) weighting in policy gradient
measuring episode returns
measuring episode lengths, how many batches of experiences took place

reset episode variables:
obs, state =first env to draw observation from starting distribution
done = boolean flag to signal from environment that the episode is over
episode rewards = list for rewards accrued throughout episode

collect a set of experiences by taking actions in an environment
with the current policy (which means no updates on the parameter theta):
step 1. 
agent gets current obs, state from the environment
uses the policy network to get logits
then based on logits compute an action distribution
from this action distribution sample one action
now you have a random sampled action taken from an action distribution
that the policy network predicted

step 2. 
this action is then received by the environment and returns a new state/obs,
the reward associated with the action taken, and whether you have reached
your end goal

remember to save history of actions taken and episode rewards

continue to perform steps 1 and 2 until you have reached the ended goal or
the environment signals that its over (how is this determined?)
for this current policy

if the environment says it's done then we collect the
batch return, and weights (discounting factor?)
if there hasn't been enough experiences in this batch this we
continue on collecting more until we have enough experience, even
when the environment has noted it is done

once we have collected enough experiences for a batch,
then we take a single policy gradient update step
by computing the loss of the policy network and backpropagating
this loss and updating the gradients in the policy network

the numbers you want to keep track are, the loss which in this case
doesn't mean much like in supervised learning, the returns which is the
sum of experience rewards for each batch of experiences
the number of batches in an episode