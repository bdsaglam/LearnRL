import numpy as np
from gym.wrappers import TransformObservation
from gym_miniworld.envs.westworld import DecoreOption, WestWorld

from spinup.wrappers import PyTorchImageWrapper


def make_env(seed, max_episode_steps=1500):
    env = WestWorld(
        seed=seed,
        room_size=2,
        obs_width=64,
        obs_height=64,
        max_episode_steps=max_episode_steps,
        decore_option=DecoreOption.PORTRAIT,
        num_chars_on_wall=1,
    )
    env = TransformObservation(env, f=lambda obs: np.asarray(obs, dtype=np.float32) / 255.0)
    env = PyTorchImageWrapper(env)
    return env
