import pathlib

import gym
import torch

from spinup import a2c
from spinup.algos.a2c import get_arg_parser
from spinup.constants import DEVICE
from spinup.examples.classic_control.core import make_model
from spinup.utils import mpi_tools
from spinup.utils.experiment_utils import get_latest_saved_file

if __name__ == '__main__':

    parser = get_arg_parser()

    parser.add_argument('--env', type=str, default='CartPole-v1')
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--allow_run_as_root', action='store_true')
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--num_hidden', type=int, default=2)
    parser.add_argument('--continue_training', '-c', action='store_true')
    parser.add_argument('--saved_model_file', '-f', type=str, default=None)

    args = parser.parse_args()

    if args.cpu > 1:
        mpi_tools.mpi_fork(args.cpu, allow_run_as_root=args.allow_run_as_root)  # run parallel code with mpi

    # Setup experiment name
    env = gym.make(args.env)

    from spinup.utils.run_utils import setup_logger_kwargs

    experiment_name = args.exp_name or env.spec.id
    logger_kwargs = setup_logger_kwargs(experiment_name, args.seed)

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
            model_kwargs=dict(hidden_sizes=[args.hidden_size] * args.num_hidden),
        )

    # Setup epoch number and action frequencies
    epochs = args.epochs // args.cpu
    save_every = args.save_every or max(10, epochs // 10)
    log_every = args.log_every or max(10, epochs // 100)
    test_every = args.test_every or max(10, epochs // 10)

    assert save_every <= epochs
    assert log_every <= epochs
    assert test_every <= epochs

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
