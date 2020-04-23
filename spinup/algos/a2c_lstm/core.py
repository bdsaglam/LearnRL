import itertools
from copy import deepcopy

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

import spinup.algos.pytorch.a2c_lstm.utils as utils


class MLPCategoricalActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation=nn.ReLU):
        super().__init__()
        self.logits_net = utils.mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def forward(self, obs):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        logits = self.logits_net(obs)
        return Categorical(logits=logits)


class MLPVFunction(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, activation=nn.ReLU):
        super().__init__()
        self.q = utils.mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return self.q(obs)


class FeatureExtractor(nn.Module):
    def __init__(self, obs_dim, feature_dim):
        super().__init__()
        self.lstmCell = nn.LSTMCell(input_size=obs_dim, hidden_size=feature_dim)

    def forward(self, obs, hidden):
        hx, cx = self.lstmCell(obs, hidden)
        return hx, cx

    def initial_hidden_state(self, batch_size=1):
        hx = torch.zeros(batch_size, self.lstmCell.hidden_size, dtype=torch.float32)
        cx = torch.zeros(batch_size, self.lstmCell.hidden_size, dtype=torch.float32)
        return hx, cx


class PolicyLSTM(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_size, activation=nn.ReLU):
        super().__init__()
        self.lstm = nn.LSTM(input_size=obs_dim, hidden_size=hidden_size)
        self.activation = activation()
        self.value_head = nn.Linear(hidden_size, 1)
        self.policy_head = nn.Linear(hidden_size, action_dim)

    def forward(self, obs):
        x = obs.unsqueeze(1)
        x, _ = self.lstm(x, self.initial_hidden_state())
        x = x.squeeze(1)
        x = self.activation(x)
        v = self.value_head(x)
        logits = self.policy_head(x)
        return v, Categorical(logits=logits)

    def initial_hidden_state(self):
        hx = torch.zeros(1, 1, self.lstm.hidden_size, dtype=torch.float32)
        cx = torch.zeros(1, 1, self.lstm.hidden_size, dtype=torch.float32)
        return hx, cx


class ActorCritic:
    def __init__(self, feature_extractor, actor_network, critic_network):
        super().__init__()
        self.fe = feature_extractor
        self.pi = actor_network

        self.v1 = critic_network
        self.v2 = deepcopy(critic_network)

    def act(self, obs, hidden=None):
        return self.act_dist(obs, hidden).sample().numpy()

    def act_deterministic(self, obs, hidden=None):
        return self.act_dist(obs, hidden).probs.argmax(-1).numpy()

    def act_dist(self, obs, hidden=None):
        with torch.no_grad():
            hidden = hidden or self.fe.initial_hidden_state()
            hx, cx = self.fe(obs.unsqueeze(0), hidden)
            feature_tensor = hx.squeeze(0)
            dist = self.pi(feature_tensor)
        return dist

    def parameters(self):
        return itertools.chain(self.fe.parameters(),
                               self.pi.parameters(),
                               self.v1.parameters(),
                               self.v2.parameters())


class TestAgent:
    def __init__(self, ac):
        super().__init__()
        self.ac = ac
        self.hidden_state = self.ac.fe.initial_hidden_state()

    def act(self, obs, hidden=None):
        return self.act_dist(obs, hidden).sample().numpy()

    def act_deterministic(self, obs, hidden=None):
        return self.act_dist(obs, hidden).probs.argmax(-1).numpy()

    def act_dist(self, obs, hidden=None):
        with torch.no_grad():
            hidden = hidden or self.hidden_state
            hx, cx = self.ac.fe(obs.unsqueeze(0), hidden)
            feature_tensor = hx.squeeze(0)
            dist = self.ac.pi(feature_tensor)
            self.hidden_state = (hx, cx)
        return dist

    def reset(self):
        self.hidden_state = self.ac.fe.initial_hidden_state()
