import pathlib

import gym
import torch

from spinup.algos.a2c import a2c
from spinup.constants import DEVICE
from spinup.examples.LSTM.core import make_model
from spinup.utils import mpi_tools
from spinup.utils.experiment_utils import get_latest_saved_file

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--env', type=str, default='CartPole-v1')
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--allow_run_as_root', action='store_true')
    parser.add_argument('--lstm_hidden_size', type=int, default=64)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--num_hidden', type=int, default=1)
    parser.add_argument('--gamma', '-g', type=float, default=0.99)
    parser.add_argument('--tau', '-t', type=float, default=0.99)
    parser.add_argument('--max_grad_norm', type=float, default=None)
    parser.add_argument('--use_gae', type=bool, default=True)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--value_loss_coef', '-vl', type=float, default=1)
    parser.add_argument('--policy_loss_coef', '-pl', type=float, default=1)
    parser.add_argument('--entropy_loss_coef', '-el', type=float, default=1)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--steps_per_epoch', type=int, default=64)
    parser.add_argument('--episode_len_limit', type=int, default=None)
    parser.add_argument('--continue_training', '-c', action='store_true')
    parser.add_argument('--saved_model_file', '-f', type=str, default=None)
    parser.add_argument('--save_every', type=int, default=None)
    parser.add_argument('--log_every', type=int, default=None)
    parser.add_argument('--test_every', type=int, default=None)
    parser.add_argument('--num_test_episodes', type=int, default=5)
    parser.add_argument('--test_episode_len_limit', type=int, default=None)
    parser.add_argument('--deterministic', '-d', action='store_true')
    parser.add_argument('--solved_score', type=int, default=None)

    args = parser.parse_args()

    # Setup experiment name
    env = gym.make(args.env)

    from spinup.utils.run_utils import setup_logger_kwargs

    experiment_name = args.exp_name or env.spec.id
    logger_kwargs = setup_logger_kwargs(experiment_name, args.seed)

    torch.set_num_threads(torch.get_num_threads())

    # Load or create model
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
        model = make_model(
            env,
            lstm_hidden_size=args.lstm_hidden_size,
            actor_hidden_sizes=[args.hidden_size] * args.num_hidden,
            critic_hidden_sizes=[args.hidden_size] * args.num_hidden,
        )

    # Setup epoch number and action frequencies
    epochs = args.epochs // args.cpu
    save_every = args.save_every or max(10, epochs // 10)
    log_every = args.log_every or max(10, epochs // 100)
    test_every = args.test_every or max(10, epochs // 10)

    assert save_every <= epochs
    assert log_every <= epochs
    assert test_every <= epochs

    if args.cpu > 1:
        mpi_tools.mpi_fork(args.cpu, allow_run_as_root=args.allow_run_as_root)  # run parallel code with mpi

    a2c(
        env_fn=lambda: gym.make(args.env),
        model=model,
        seed=args.seed,
        num_cpu=args.cpu,
        device=DEVICE,
        gamma=args.gamma,
        use_gae=args.use_gae,
        tau=args.tau,
        max_grad_norm=args.max_grad_norm,
        learning_rate=args.learning_rate,
        value_loss_coef=args.value_loss_coef,
        policy_loss_coef=args.policy_loss_coef,
        entropy_loss_coef=args.entropy_loss_coef,
        epochs=epochs,
        steps_per_epoch=args.steps_per_epoch,
        episode_len_limit=args.episode_len_limit,
        save_every=save_every,
        log_every=log_every,
        logger_kwargs=logger_kwargs,
        test_every=test_every,
        num_test_episodes=args.num_test_episodes,
        test_episode_len_limit=args.test_episode_len_limit,
        deterministic=args.deterministic,
        solved_score=args.solved_score,
    )
