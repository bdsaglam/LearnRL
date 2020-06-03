from spinup.constants import DEVICE
from spinup.examples.mmi706.environment import make_env
from spinup.examples.mmi706.evaluate import evaluate_agent
from spinup.utils.experiment_utils import load_env_and_model

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_folder', type=str)
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--non_deterministic', '-nd', action='store_true')
    parser.add_argument('--num_episodes', '-n', type=int, default=3)
    parser.add_argument('--episode_len_limit', '-l', type=int, default=None)
    parser.add_argument('--norender', '-nr', action='store_true')
    args = parser.parse_args()

    print("Device: ", DEVICE)

    env, agent = load_env_and_model(args.experiment_folder,
                                    args.itr if args.itr >= 0 else 'last')
    if env is None:
        env = make_env(42, 100)

    evaluate_agent(env,
                   agent,
                   deterministic=not args.non_deterministic,
                   num_episodes=args.num_episodes,
                   render=not args.norender)
