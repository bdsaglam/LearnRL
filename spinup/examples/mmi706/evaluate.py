import time

import torch

from spinup.examples.mmi706.agent import Agent
from spinup.utils.logx import EpochLogger


def evaluate_agent(
        env_fn,
        agent: Agent,
        deterministic=True,
        num_episodes=5,
        render=False,
        logger=None
):
    assert env_fn is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :( \n\n Check out the readthedocs " + \
        "page on Experiment Outputs for how to handle this situation."

    env = env_fn()
    assert env.spec.max_episode_steps > 0

    if logger is None:
        logger = EpochLogger()

    episode_info = []
    goal_grid_code = None
    for _ in range(num_episodes):
        obs = env.reset()
        agent.reset()
        reward = 0

        done = False
        episode_return = 0
        t = 0
        while not done:
            if render:
                env.render()
                time.sleep(1e-3)

            with torch.no_grad():
                action = agent.act(obs, reward, goal_grid_code, deterministic)
            obs, reward, done, _ = env.step(action)
            episode_return += reward
            t += 1

        if done:
            goal_grid_code = agent.current_grid_code.detach().cpu().numpy()

        episode_info.append((t, episode_return))
        logger.store(TestEpRet=episode_return, TestEpLen=t)

    logger.log_tabular('EpisodeLimit', env.spec.max_episode_steps)
    logger.log_tabular('TestEpLen', with_min_and_max=True)
    logger.log_tabular('TestEpRet', with_min_and_max=True)
    logger.dump_tabular()

    env.close()

    return episode_info
