import gym
import numpy as np


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
