import gym

from spinup.core.api import IAgent
from spinup.utils.evaluate import evaluate_agent


class RandomAgent(IAgent):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, *args, **kwargs):
        return self.action_space.sample()

    def reset(self):
        pass


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Breakout-v0')
    parser.add_argument('--num_episodes', '-n', type=int, default=5)
    parser.add_argument('--episode_len_limit', '-l', type=int, default=None)
    parser.add_argument('--norender', '-nr', action='store_true')
    args = parser.parse_args()

    env = gym.make(args.env)
    agent = RandomAgent(env.action_space)
    evaluate_agent(env,
                   agent,
                   num_episodes=args.num_episodes,
                   episode_len_limit=args.episode_len_limit,
                   render=not args.norender)
