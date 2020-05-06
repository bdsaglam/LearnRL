import numpy as np
from gym.wrappers import TransformObservation
from gym_miniworld.envs.westworld import DecoreOption, WestWorld

from spinup.wrappers import PyTorchImageWrapper


def make_env(seed, max_episode_steps=1500):
    env = WestWorld(
        seed=seed,
        obs_width=128,
        obs_height=128,
        max_episode_steps=max_episode_steps,
        decore_option=DecoreOption.DIGIT,
        num_chars_on_wall=2,
    )
    env = TransformObservation(env, f=lambda obs: np.asarray(obs, dtype=np.float32) / 255.0)
    env = PyTorchImageWrapper(env)
    return env
