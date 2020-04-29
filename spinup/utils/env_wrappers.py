import gym
import numpy as np
from gym.spaces.box import Box


class PyTorchImageWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            low=self.observation_space.low[0, 0, 0],
            high=self.observation_space.high[0, 0, 0],
            shape=[obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return observation.transpose(2, 0, 1)


class CropImageWrapper(gym.ObservationWrapper):
    def __init__(self, env, row1=0, row2=None, col1=0, col2=None):
        super().__init__(env)
        obs_shape = list(self.observation_space.shape)
        h, w = obs_shape[0], obs_shape[1]

        self.row1 = row1
        self.row2 = row2 or h
        self.col1 = col1
        self.col2 = col2 or w
        obs_shape[0] = self.row2 - self.row1
        obs_shape[1] = self.col2 - self.col1

        observation_space = env.observation_space
        observation_space.shape = obs_shape
        self.observation_space = observation_space

    def observation(self, observation):
        return observation[self.row1:self.row2, self.col1:self.col2]


class ConcatWrapper(gym.ObservationWrapper):
    def __init__(self, env, axis):
        super().__init__(env)
        self.axis = axis

        new_obs_shape = np.concatenate(np.empty(self.observation_space.shape), axis=axis).shape

        observation_space = env.observation_space
        observation_space.shape = new_obs_shape
        self.observation_space = observation_space

    def observation(self, observation):
        return np.concatenate(observation, axis=self.axis)
