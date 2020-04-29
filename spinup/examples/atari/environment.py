import gym
import numpy as np
from gym.wrappers import AtariPreprocessing, FrameStack, TransformObservation

from spinup.utils.atari_wrappers import FireReset
from spinup.utils.env_wrappers import PyTorchImageWrapper, ConcatWrapper


def make_atari_env(env_name):
    env = gym.make(env_name)
    env = FireReset(env)
    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=2,
        screen_size=84,
        terminal_on_life_loss=False,
        grayscale_obs=False,
        scale_obs=False
    )
    env = PyTorchImageWrapper(env)
    env = FrameStack(env, num_stack=4)
    env = TransformObservation(env, f=np.array)
    env = ConcatWrapper(env, axis=0)
    env = TransformObservation(env, f=lambda obs: np.asarray(obs, dtype=np.float32) / 255.0)

    return env
