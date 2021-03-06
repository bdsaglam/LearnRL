import gym
from gym.spaces.box import Box


class PyTorchImageWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape
        h, w = obs_shape[0], obs_shape[1]
        if len(obs_shape) == 3:
            low = self.observation_space.low[0, 0, 0]
            high = self.observation_space.high[0, 0, 0]
            c = obs_shape[2]
        else:
            low = self.observation_space.low[0, 0]
            high = self.observation_space.high[0, 0]
            c = 1

        self.observation_space = Box(
            low=low,
            high=high,
            shape=[c, h, w],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        if len(observation.shape) == 3:
            return observation.transpose(2, 0, 1)

        return observation[None, :, :]


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
