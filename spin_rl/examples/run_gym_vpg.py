from spinup import vpg_pytorch
from spinup.utils.run_utils import ExperimentGrid
import torch


if __name__ == '__main__':
    grid = ExperimentGrid(name='vpg-torch-cart-bench')
    grid.add('env_name', 'CartPole-v0')
    grid.add('seed', [0])
    grid.add('epochs', 2)
    grid.add('steps_per_epoch', 100)
    grid.add('gamma', [0, 0.5, 1])
    grid.add('ac_kwargs:hidden_sizes', [(32,), (64, 64)], 'hid')
    grid.add('ac_kwargs:activation', [torch.nn.Tanh], '')

    grid.run(vpg_pytorch, num_cpu=4)
