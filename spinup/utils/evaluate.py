import time

import torch

from spinup.core.api import IAgent
from spinup.utils.logx import EpochLogger


def evaluate_agent(
        env,
        agent: IAgent,
        deterministic=True,
        num_episodes=5,
        episode_len_limit=None,
        render=False,
        logger=None
):
    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :( \n\n Check out the readthedocs " + \
        "page on Experiment Outputs for how to handle this situation."

    if episode_len_limit is None:
        if env.unwrapped.spec and env.unwrapped.spec.max_episode_steps:
            episode_len_limit = env.spec.max_episode_steps
        else:
            raise ValueError("Episode length limit must be specified")

    if logger is None:
        logger = EpochLogger()

    episode_info = []
    for _ in range(num_episodes):
        obs = env.reset()
        agent.reset()
        done = False
        episode_return = 0
        t = 0
        while not done and t != episode_len_limit:
            if render:
                env.render()
                time.sleep(1e-3)

            with torch.no_grad():
                action = agent.act(obs, deterministic)
            obs, reward, done, _ = env.step(action)
            episode_return += reward
            t += 1
        episode_info.append((t, episode_return))
        logger.store(TestEpRet=episode_return, TestEpLen=t)

    logger.log_tabular('EpisodeLimit', episode_len_limit)
    logger.log_tabular('TestEpLen', with_min_and_max=True)
    logger.log_tabular('TestEpRet', with_min_and_max=True)
    logger.dump_tabular()

    return episode_info
