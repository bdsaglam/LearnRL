import pathlib

import gym
import torch

from spinup.algos.a2c.train import train
from spinup.constants import DEVICE
from spinup.examples.atari import core
from spinup.utils.experiment_utils import get_latest_saved_file

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--env', type=str, default='Breakout-v0')
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--num_hidden', type=int, default=1)
    parser.add_argument('--gamma', '-g', type=float, default=0.99)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--value_loss_coef', '-vl', type=float, default=1)
    parser.add_argument('--policy_loss_coef', '-pl', type=float, default=1)
    parser.add_argument('--entropy_loss_coef', '-el', type=float, default=1)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--steps_per_epoch', type=int, default=10)
    parser.add_argument('--episode_len_limit', type=int, default=1000)
    parser.add_argument('--continue_training', '-c', action='store_true')
    parser.add_argument('--saved_model_file', '-f', type=str, default=None)
    parser.add_argument('--save_every', type=int, default=None)
    parser.add_argument('--log_every', type=int, default=None)
    parser.add_argument('--test_every', type=int, default=None)
    parser.add_argument('--num_test_episodes', type=int, default=5)
    parser.add_argument('--test_episode_len_limit', type=int, default=None)
    parser.add_argument('--deterministic', '-d', action='store_true')
    parser.add_argument('--solve_score', type=int, default=None)

    args = parser.parse_args()

    env = gym.make(args.env)

    from spinup.utils.run_utils import setup_logger_kwargs

    experiment_name = args.exp_name or env.spec.id
    logger_kwargs = setup_logger_kwargs(experiment_name, args.seed)

    torch.set_num_threads(torch.get_num_threads())

    saved_model_file = None
    if args.saved_model_file:
        saved_model_file = pathlib.Path(args.saved_model_file)
    elif args.continue_training:
        save_dir = pathlib.Path(logger_kwargs['output_dir'], 'pyt_save')
        if save_dir.exists():
            saved_model_file = get_latest_saved_file(save_dir, prefix='model')

    if saved_model_file:
        assert saved_model_file.exists()
        model = torch.load(saved_model_file, map_location=DEVICE)
        for p in model.parameters():
            p.requires_grad_()
        print("Loaded model from: ", saved_model_file)
    else:
        model = core.make_model(
            env,
            model_kwargs=dict(hidden_sizes=[args.hidden_size] * args.num_hidden),
        )

    epochs = args.epochs
    save_every = args.save_every or max(10, epochs // 10)
    log_every = args.log_every or max(10, epochs // 100)
    test_every = args.test_every or max(10, epochs // 10)

    assert save_every <= epochs
    assert log_every <= epochs
    assert test_every <= epochs

    train(
        env=env,
        test_env=gym.make(args.env),
        model=model,
        seed=args.seed,
        device=DEVICE,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        value_loss_coef=args.value_loss_coef,
        policy_loss_coef=args.policy_loss_coef,
        entropy_loss_coef=args.entropy_loss_coef,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        episode_len_limit=args.episode_len_limit,
        save_every=save_every,
        log_every=log_every,
        logger_kwargs=logger_kwargs,
        test_every=test_every,
        num_test_episodes=args.num_test_episodes,
        test_episode_len_limit=args.test_episode_len_limit,
        deterministic=args.deterministic,
        solve_score=args.solve_score,
    )
