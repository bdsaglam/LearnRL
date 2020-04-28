from copy import deepcopy

import torch
import torch.nn as nn

from spinup.constants import DEVICE
from spinup.core.api import IActorCritic
from spinup.core.approximators import MLPCategoricalActor, MLPVFunction
from spinup.core.bellman import generalized_advantage_estimate, calculate_returns
from spinup.utils import nn_utils


class History:
    def __init__(self):
        self.log_probs = []
        self.entropy = []
        self.v1 = []
        self.v2 = []

    def store(self, log_prob, entropy, v1, v2):
        self.log_probs.append(log_prob)
        self.entropy.append(entropy)
        self.v1.append(v1)
        self.v2.append(v2)

    def data(self):
        batch_log_probs = torch.cat(self.log_probs, 0)
        batch_entropy = torch.cat(self.entropy, 0)
        batch_v1 = torch.cat(self.v1, 0)
        batch_v2 = torch.cat(self.v2, 0)
        return batch_log_probs, batch_entropy, batch_v1, batch_v2


class ActorCritic(IActorCritic):
    def __init__(self, feature_extractor: nn.Module, actor: nn.Module, critic: nn.Module):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.actor = actor
        self.critic1 = critic
        self.critic2 = deepcopy(critic)

        self.train_history = History()

    def reset(self):
        self.train_history = History()

    def step(self, obs_tensor):
        batch_obs = obs_tensor.unsqueeze(0)
        feature_tensor = self.feature_extractor(batch_obs)
        v1 = self.critic1(feature_tensor)
        v2 = self.critic2(feature_tensor)
        dist = self.actor(feature_tensor)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        self.train_history.store(
            log_prob=log_prob,
            entropy=entropy,
            v1=v1,
            v2=v2,
        )

        return action.squeeze(0)

    def predict_value(self, obs_tensor):
        batch_obs = obs_tensor.unsqueeze(0)
        with torch.no_grad():
            feature_tensor = self.feature_extractor(batch_obs)
            v1 = self.critic1(feature_tensor)
            v2 = self.critic2(feature_tensor)
            v = torch.min(v1, v2)
        return v.squeeze(0)

    def compute_loss(self,
                     rewards,
                     dones,
                     next_value,
                     discount_factor,
                     use_gae=True,
                     tau=0.95,
                     value_loss_coef=1,
                     policy_loss_coef=1,
                     entropy_reg_coef=1
                     ):
        returns = calculate_returns(rewards=rewards,
                                    next_value=next_value,
                                    discount_factor=discount_factor)
        batch_return = torch.tensor(returns, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        # all tensors have shape of (T, 1)
        # MSE loss against Bellman backup
        batch_log_probs, batch_entropy, batch_v1, batch_v2 = self.train_history.data()
        loss_v1 = (batch_return - batch_v1).pow(2).mean()
        loss_v2 = (batch_return - batch_v2).pow(2).mean()
        loss_v = value_loss_coef * (loss_v1 + loss_v2)

        # Policy loss
        batch_value = torch.min(batch_v1.detach(), batch_v2.detach())
        if use_gae:
            advantages = generalized_advantage_estimate(rewards=rewards,
                                                        values=batch_value.squeeze(0).cpu().numpy(),
                                                        next_value=next_value,
                                                        discount_factor=discount_factor,
                                                        tau=tau)
            batch_advantage = torch.tensor(advantages, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        else:
            batch_advantage = batch_return - batch_value
        loss_pi = -policy_loss_coef * (batch_advantage * batch_log_probs).mean()

        # Entropy-regularization
        loss_entropy = -entropy_reg_coef * batch_entropy.mean()

        # total loss
        loss = loss_v + loss_pi + loss_entropy

        # Useful info for logging
        info = dict(
            LossV=loss_v.detach().cpu().numpy(),
            LossPi=loss_pi.detach().cpu().numpy(),
            LossEntropy=loss_entropy.detach().cpu().numpy(),
            V1Vals=batch_v1.detach().cpu().numpy(),
            V2Vals=batch_v2.detach().cpu().numpy(),
            LogPi=batch_log_probs.detach().cpu().numpy(),
        )

        return loss, info

    def act(self, obs, deterministic=False):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            feature = self.feature_extractor(obs_tensor)
            dist = self.actor(feature)
            if deterministic:
                action = dist.probs.argmax(-1)
            else:
                action = dist.sample()

        return action.cpu().squeeze(0).numpy()


def make_model(env, model_kwargs):
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    # Create actor-critic module and target networks
    feature_dim = obs_dim
    feature_extractor = torch.nn.Identity()
    actor_network = MLPCategoricalActor(feature_dim, act_dim, **model_kwargs)
    critic_network = MLPVFunction(feature_dim, **model_kwargs)
    ac = ActorCritic(
        feature_extractor=feature_extractor,
        actor=actor_network,
        critic=critic_network
    )

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(nn_utils.count_vars(module) for module in [actor_network, critic_network])
    print('\nNumber of parameters: \t actor: %d, \t critic: %d\n' % var_counts)

    return ac
