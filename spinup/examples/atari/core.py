import itertools
from copy import deepcopy

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from spinup.constants import DEVICE
from spinup.core.api import IAgent, IActorCritic
from spinup.examples.atari.feature_extraction import make_frame_buffer, preprocess
from spinup.utils.nn_utils import mlp


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


class ActorCritic(nn.Module, IActorCritic):
    def __init__(self, feature_extractor: nn.Module, actor: nn.Module, critic: nn.Module):
        super().__init__()
        self.feature_extractor = feature_extractor
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

    def critic_parameters(self):
        return itertools.chain(self.critic1.parameters(), self.critic2.parameters())

    def actor_parameters(self):
        return self.actor.parameters()

    def feature_extractor_parameters(self):
        return self.feature_extractor.parameters()

    def compute_loss(self, features, log_probs, entropies, returns, value_loss_coef, entropy_reg_coef):
        v1 = self.critic1(features)
        v2 = self.critic2(features)

        # MSE loss against Bellman backup
        loss_v1 = (returns - v1).pow(2).mean()
        loss_v2 = (returns - v2).pow(2).mean()
        loss_v = value_loss_coef * (loss_v1 + loss_v2)

        # Useful info for logging
        v_info = dict(V1Vals=v1.cpu().detach().numpy(),
                      V2Vals=v2.cpu().detach().numpy())

        values = torch.min(v1.detach(), v2.detach())
        advantages = returns - values

        # Entropy-regularized policy loss
        loss_pi = -entropy_reg_coef * entropies.mean() - (advantages * log_probs).mean()

        # Useful info for logging
        pi_info = dict(LogPi=log_probs.cpu().detach().numpy(),
                       MeanEntropy=entropies.cpu().detach().mean().numpy())

        return loss_v, v_info, loss_pi, pi_info


class Agent(IAgent):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor_critic = actor_critic.to(DEVICE)
        self.buffer = make_frame_buffer()

    def reset(self):
        self.buffer = make_frame_buffer()

    def act(self, obs, deterministic=False):
        dist = self.action_dist(obs)
        if deterministic:
            return dist.probs.argmax(-1).cpu().numpy()
        return dist.sample().cpu().numpy()

    def action_dist(self, obs):
        image_tensor = preprocess(obs)
        self.buffer.append(image_tensor)
        batch_input = torch.cat(list(self.buffer), dim=0).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            feature_tensor = self.actor_critic.feature_extractor(batch_input).squeeze(0)
            dist = self.actor_critic.infer_action_dist(feature_tensor)
        return dist
