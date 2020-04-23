import itertools
from copy import deepcopy

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from spinup.constants import DEVICE
from spinup.core.api import IAgent, IActorCritic
from spinup.utils import nn_utils


class MLPCategoricalActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation=nn.ReLU):
        super().__init__()
        self.logits_net = nn_utils.mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def forward(self, obs):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        logits = self.logits_net(obs)
        return Categorical(logits=logits)


class MLPVFunction(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation=nn.ReLU):
        super().__init__()
        self.q = nn_utils.mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return self.q(obs)


class ActorCritic(nn.Module, IActorCritic):
    def __init__(self, actor: nn.Module, critic: nn.Module):
        super().__init__()
        self.actor = actor
        self.critic1 = critic
        self.critic2 = deepcopy(critic)

    def infer_value(self, feature_tensor):
        v1 = self.critic1(feature_tensor)
        v2 = self.critic2(feature_tensor)
        values = torch.min(v1, v2)
        return values

    def infer_action_dist(self, feature_tensor):
        dist = self.actor(feature_tensor)
        return dist

    def infer_value_action_dist(self, feature_tensor):
        return self.infer_value(feature_tensor), self.infer_action_dist(feature_tensor)

    def critic_parameters(self):
        return itertools.chain(self.critic1.parameters(), self.critic2.parameters())

    def actor_parameters(self):
        return self.actor.parameters()

    def compute_loss(self, features, log_probs, entropies, returns, entropy_reg_coef):
        v1 = self.critic1(features)
        v2 = self.critic2(features)

        # MSE loss against Bellman backup
        loss_v1 = (returns - v1).pow(2).mean()
        loss_v2 = (returns - v2).pow(2).mean()
        loss_v = loss_v1 + loss_v2

        # Useful info for logging
        v_info = dict(V1Vals=v1.cpu().detach().numpy(),
                      V2Vals=v2.cpu().detach().numpy())

        values = torch.min(v1.detach(), v2.detach())
        advantages = returns - values

        # Entropy-regularized policy loss
        loss_pi = 1 * (-entropy_reg_coef * entropies.mean() - (advantages * log_probs).mean())

        # Useful info for logging
        pi_info = dict(LogPi=log_probs.cpu().detach().numpy(),
                       MeanEntropy=entropies.cpu().detach().mean().numpy())

        return loss_v, v_info, loss_pi, pi_info


class Agent(IAgent):
    def __init__(self, actor: [nn.Module]):
        super().__init__()
        self.actor = actor.to(DEVICE)

    def reset(self):
        pass

    def act(self, obs, deterministic=False):
        feature_tensor = torch.tensor(obs, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            dist = self.actor(feature_tensor)
            if deterministic:
                action = dist.probs.argmax(-1)
            else:
                action = dist.sample()

        return action.cpu().numpy()
