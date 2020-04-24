import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from spinup.constants import DEVICE
from spinup.core.api import IAgent
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


class Model(nn.Module):
    def __init__(self, feature_extractor: nn.Module, actor: nn.Module, critic: nn.Module):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.actor = actor
        self.critic1 = critic
        # self.critic2 = deepcopy(critic)

    def step(self, obs):
        feature = self.feature_extractor(obs)
        value = self.critic1(feature)

        dist = self.actor(feature)
        return value, dist

    def compute_loss(self, values, log_probs, entropies, returns, value_loss_coef, entropy_reg_coef):
        # MSE loss against Bellman backup
        loss_v = value_loss_coef * (returns - values).pow(2).mean()

        # Entropy-regularized policy loss
        advantages = returns - values
        loss_pi = 1 * (-entropy_reg_coef * entropies.mean() - (advantages * log_probs).mean())

        return loss_v, loss_pi

    def critic_parameters(self):
        return self.critic1.parameters()
        # return itertools.chain(self.critic1.parameters(), self.critic2.parameters())

    def actor_parameters(self):
        return self.actor.parameters()

    def feature_extractor_parameters(self):
        return self.feature_extractor.parameters()


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
