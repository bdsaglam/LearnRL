import time

import torch

from spinup import EpochLogger
from spinup.core.api import IAgent


def evaluate_agent(
        env,
        agent: IAgent,
        deterministic=False,
        num_episodes=5,
        episode_len_limit=None,
        render=False,
        logger=EpochLogger()
):
    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :( \n\n Check out the readthedocs " + \
        "page on Experiment Outputs for how to handle this situation."

    if episode_len_limit is None:
        if env.spec and env.spec.max_episode_steps:
            episode_len_limit = env.spec.max_episode_steps
        else:
            episode_len_limit = 1000

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
        logger.store(TestEpRet=episode_return, TestEpLen=t)

    logger.log_tabular('TestEpRet', with_min_and_max=True)
    logger.log_tabular('TestEpLen', with_min_and_max=True)
    logger.dump_tabular()
