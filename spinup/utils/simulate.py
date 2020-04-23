from spinup.utils.evaluate import evaluate_agent
from spinup.utils.experiment_utils import load_env_and_agent

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_folder', type=str)
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--deterministic', '-d', action='store_true')
    parser.add_argument('--num_episodes', '-n', type=int, default=5)
    parser.add_argument('--episode_len_limit', '-l', type=int, default=None)
    parser.add_argument('--norender', '-nr', action='store_true')
    args = parser.parse_args()

    env, agent = load_env_and_agent(args.experiment_folder,
                                    args.itr if args.itr >= 0 else 'last')

    evaluate_agent(env,
                   agent,
                   deterministic=args.deterministic,
                   num_episodes=args.num_episodes,
                   episode_len_limit=args.episode_len_limit,
                   render=not args.norender)
