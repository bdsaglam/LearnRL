from collections import deque
from copy import deepcopy

import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torch.distributions.categorical import Categorical

FRAME_SHAPE = (3, 80, 80)
NUM_RECENT_FRAMES = 4


def preprocess(obs):
    img = Image.fromarray(obs[32:192])
    c, h, w = FRAME_SHAPE
    transformation = torchvision.transforms.Compose([
        torchvision.transforms.Resize((h, w)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0, 0, 0), (255, 255, 255))
    ])
    img_tensor = transformation(img)
    return img_tensor


def make_frame_buffer():
    c, h, w = FRAME_SHAPE
    buffer = deque(maxlen=NUM_RECENT_FRAMES)
    for _ in range(NUM_RECENT_FRAMES - 1):
        t = torch.zeros(c, h, w, dtype=torch.float32)
        buffer.append(t)

    return buffer


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


def frames_feature_extractor():
    c, h, w = FRAME_SHAPE

    network = nn.Sequential(
        nn.Conv2d(c * NUM_RECENT_FRAMES, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(64, 128, kernel_size=7, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Flatten()
    )

    with torch.no_grad():
        feature_dim = network(torch.rand(1, c * NUM_RECENT_FRAMES, h, w)).shape

    return network, feature_dim[-1]


class ActorCritic(nn.Module):
    def __init__(self, feature_extractor, actor_network, critic_network):
        super().__init__()
        self.fe = feature_extractor
        self.pi = actor_network
        self.v1 = critic_network
        self.v2 = deepcopy(critic_network)

    def compute_loss(self, features, log_probs, entropies, returns, entropy_reg_coef):
        v1 = self.v1(features)
        v2 = self.v2(features)

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


class Agent:
    def __init__(self, ac, device=torch.device("cpu")):
        super().__init__()
        self.ac = ac.to(device)
        self.device = device
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
        batch_input = torch.cat(list(self.buffer), dim=0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feature_tensor = self.ac.fe(batch_input).squeeze(0)
            dist = self.ac.pi(feature_tensor)
        return dist
