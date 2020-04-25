import torch.nn as nn
from torch.distributions.categorical import Categorical


class MLPCategoricalActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation=nn.ReLU):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def forward(self, obs):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        logits = self.logits_net(obs)
        return Categorical(logits=logits)


class MLPVFunction(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation=nn.ReLU):
        super().__init__()
        self.q = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return self.q(obs)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)
