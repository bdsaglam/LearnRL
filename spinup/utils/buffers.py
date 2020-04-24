import numpy as np
import torch

from spinup.utils import general_utils


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, capacity):
        self.capacity = capacity

        self.obs_buf = np.zeros(general_utils.combined_shape(capacity, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(general_utils.combined_shape(capacity, obs_dim), dtype=np.float32)
        self.action_buf = np.zeros(general_utils.combined_shape(capacity, act_dim), dtype=np.float32)
        self.reward_buf = np.zeros((capacity, 1), dtype=np.float32)
        self.done_buf = np.zeros((capacity, 1), dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def store(self, obs, action, reward, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.action_buf[self.ptr] = action
        self.reward_buf[self.ptr] = reward
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def data(self):
        batch = dict(
            obs=self.obs_buf[:self.ptr],
            action=self.action_buf[:self.ptr],
            reward=self.reward_buf[:self.ptr],
            obs2=self.obs2_buf[:self.ptr],
            done=self.done_buf[:self.ptr],
        )
        return {k: torch.as_tensor(v) for k, v in batch.items()}


class EpisodeBuffer:
    def __init__(self):
        self.observations = []
        self.next_observations = []
        self.actions = []
        self.rewards = []
        self.dones = []

    def store(self, observation, action, reward, next_observation, done):
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.next_observations.append(next_observation)
