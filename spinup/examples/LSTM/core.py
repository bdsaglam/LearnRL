from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn

from spinup.constants import DEVICE
from spinup.core.api import IActorCritic
from spinup.core.approximators import MLPCategoricalActor, MLPVFunction
from spinup.core.bellman import generalized_advantage_estimate, calculate_returns
from spinup.utils import nn_utils


class TrainBuffer:
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


class LSTMActorCritic(IActorCritic):
    def __init__(self, feature_extractor: nn.Module, lstm_cell: nn.LSTMCell, actor: nn.Module, critic: nn.Module):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.lstm_cell = lstm_cell
        self.actor = actor
        self.critic1 = critic
        self.critic2 = deepcopy(critic)

        self.reset()
        self.reset_for_training()

    def reset_for_training(self):
        self.train_buffer = TrainBuffer()
        self.hx = self.hx.detach()
        self.cx = self.cx.detach()

    def step(self, obs_tensor):
        batch_obs = obs_tensor.unsqueeze(0)
        feature_tensor = self.feature_extractor(batch_obs)
        self.hx, self.cx = self.lstm_cell(feature_tensor, (self.hx, self.cx))
        v1 = self.critic1(self.hx)
        v2 = self.critic2(self.hx)
        dist = self.actor(self.hx)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        self.train_buffer.store(
            log_prob=log_prob,
            entropy=entropy,
            v1=v1,
            v2=v2,
        )

        return action.squeeze(0)

    def initial_context(self):
        hx = torch.zeros(1, self.lstm_cell.hidden_size, dtype=torch.float32)
        cx = torch.zeros(1, self.lstm_cell.hidden_size, dtype=torch.float32)
        return hx, cx

    def get_context(self) -> Any:
        return self.hx, self.cx

    def set_context(self, context) -> Any:
        self.hx, self.cx = context

    def predict_value(self, obs_tensor, context):
        hx, cx = context
        with torch.no_grad():
            batch_obs = obs_tensor.unsqueeze(0)
            feature_tensor = self.feature_extractor(batch_obs)
            hx, cx = self.lstm_cell(feature_tensor, (hx, cx))
            v1 = self.critic1(hx)
            v2 = self.critic2(hx)
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
        batch_log_probs, batch_entropy, batch_v1, batch_v2 = self.train_buffer.data()
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

    def reset(self):
        self.hx = torch.zeros(1, self.lstm_cell.hidden_size, dtype=torch.float32)
        self.cx = torch.zeros(1, self.lstm_cell.hidden_size, dtype=torch.float32)

    def act(self, obs, deterministic=False):
        batch_obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            feature_tensor = self.feature_extractor(batch_obs)
            self.hx, self.cx = self.lstm_cell(feature_tensor, (self.hx, self.cx))
            dist = self.actor(self.hx)
            if deterministic:
                action = dist.probs.argmax(-1)
            else:
                action = dist.sample()

        return action.cpu().squeeze(0).numpy()


def make_model(env, lstm_hidden_size, actor_hidden_sizes, critic_hidden_sizes):
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    # Create actor-critic module and target networks
    feature_extractor = torch.nn.Identity()
    feature_dim = obs_dim
    lstm_cell = nn.LSTMCell(feature_dim, hidden_size=lstm_hidden_size)
    actor_network = MLPCategoricalActor(lstm_hidden_size, act_dim, hidden_sizes=actor_hidden_sizes)
    critic_network = MLPVFunction(lstm_hidden_size, hidden_sizes=critic_hidden_sizes)
    ac = LSTMActorCritic(
        feature_extractor=feature_extractor,
        lstm_cell=lstm_cell,
        actor=actor_network,
        critic=critic_network
    )

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(nn_utils.count_vars(module) for module in [actor_network, critic_network])
    print('\nNumber of parameters: \t actor: %d, \t critic: %d\n' % var_counts)

    return ac
