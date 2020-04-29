import pathlib
import time
from copy import deepcopy

import gym
import numpy as np
import torch
from torch.optim import Adam

from spinup.constants import DEVICE
from spinup.core.api import IActorCritic
from spinup.utils import mpi_tools, mpi_pytorch
from spinup.utils.buffers import EpisodeBuffer
from spinup.utils.evaluate import evaluate_agent
from spinup.utils.experiment_utils import get_latest_saved_file
from spinup.utils.logx import EpochLogger


def train(env_fn,
          model: IActorCritic,
          seed=0,
          num_cpu=1,
          device=torch.device("cpu"),
          epochs=1000,
          steps_per_epoch=10,
          episode_len_limit=1000,
          gamma=0.99,
          use_gae=True,
          tau=0.95,
          max_grad_norm=None,
          polyak=0.995,
          learning_rate=1e-3,
          value_loss_coef=1,
          policy_loss_coef=1,
          entropy_loss_coef=1,
          save_every=100,
          log_every=10,
          logger_kwargs=dict(),
          test_every=100,
          num_test_episodes=5,
          test_episode_len_limit=None,
          deterministic=False,
          save_freq=1,
          solve_score=None,
          ):
    use_MPI = num_cpu > 1

    if use_MPI:
        # Special function to avoid certain slowdowns from PyTorch + MPI combo.
        mpi_pytorch.setup_pytorch_for_mpi()

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    config = locals()
    del config['env_fn']
    del config['model']
    del config['logger']
    logger.save_config(config)

    # Random seed
    if use_MPI:
        seed += 10000 * mpi_tools.proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    env = env_fn()
    test_env = env_fn()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    # training model and target model
    actor_critic = model
    target_actor_critic = deepcopy(actor_critic)
    if use_MPI:
        # Sync params across processes
        mpi_pytorch.sync_params(actor_critic)
        mpi_pytorch.sync_params(target_actor_critic)
    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in target_actor_critic.parameters():
        p.requires_grad = False
    # Utilize GPU
    actor_critic.to(device)
    target_actor_critic.to(device)

    # Set up optimizers for policy and q-function
    optimizer = Adam(actor_critic.parameters(), lr=learning_rate)

    # Set up model saving
    logger.setup_pytorch_saver(actor_critic, name='model')

    def update(episode_buffer):
        # Update
        if episode_buffer.dones[-1]:
            next_value = 0.0
        else:
            last_obs = episode_buffer.next_observations[-1]
            last_obs_tensor = torch.tensor(last_obs, dtype=torch.float32).to(device)
            next_value = target_actor_critic.predict_value(last_obs_tensor).cpu().item()

        # Super critical!!
        optimizer.zero_grad()

        # Compute value and policy losses
        loss, info = actor_critic.compute_loss(rewards=np.array(episode_buffer.rewards),
                                               dones=np.array(episode_buffer.dones),
                                               next_value=next_value,
                                               discount_factor=gamma,
                                               use_gae=use_gae,
                                               tau=tau,
                                               value_loss_coef=value_loss_coef,
                                               policy_loss_coef=policy_loss_coef,
                                               entropy_reg_coef=entropy_loss_coef)
        loss.backward()
        if use_MPI:
            mpi_pytorch.mpi_avg_grads(actor_critic)

        # Optimize
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), max_grad_norm)
        optimizer.step()

        # Log losses and info
        logger.store(**info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(actor_critic.parameters(), target_actor_critic.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)
        if use_MPI:
            mpi_pytorch.sync_params(target_actor_critic)

    # Prepare for interaction with environment
    start_time = time.time()
    # Main loop: collect experience in env and update/log each epoch
    total_steps = 0
    # Reset env
    obs = env.reset()
    # Reset episode stats
    episode_return = 0
    episode_length = 0
    for _ in range(5):
        logger.store(EpRet=0, EpLen=0)
    for epoch in range(1, epochs + 1):
        actor_critic.reset()
        train_buffer = EpisodeBuffer()
        for t in range(steps_per_epoch):
            total_steps += 1

            # Get action from the model
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
            action = actor_critic.step(obs_tensor)

            # Step the env
            obs2, reward, done, _ = env.step(action.detach().cpu().item())
            episode_return += reward
            episode_length += 1

            # Store experience to buffer
            train_buffer.store(observation=obs, action=action, reward=reward, done=done, next_observation=obs2)

            # Super critical, easy to overlook step: make sure to update
            # most recent observation!
            obs = obs2

            # End of trajectory handling
            if done or episode_length > episode_len_limit:
                logger.store(EpRet=episode_return, EpLen=episode_length)
                # Reset env
                obs = env.reset()
                # Reset episode stats
                episode_return = 0
                episode_length = 0
                break

        update(train_buffer)

        # End of epoch handling
        if epoch % log_every == 0:
            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('V1Vals', average_only=True)
            logger.log_tabular('V2Vals', average_only=True)
            logger.log_tabular('LogPi', with_min_and_max=True)
            logger.log_tabular('LossV', average_only=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossEntropy', average_only=True)
            logger.log_tabular('TotalEnvInteracts', total_steps * num_cpu)
            logger.log_tabular('Time', time.time() - start_time)
            logger.dump_tabular()

        # Test agent
        solved = False
        if epoch % test_every == 0:
            # Test the performance of the deterministic version of the agent.
            actor_critic.eval()
            episode_info = evaluate_agent(env=test_env,
                                          agent=actor_critic,
                                          deterministic=deterministic,
                                          num_episodes=num_test_episodes,
                                          episode_len_limit=test_episode_len_limit,
                                          render=False)
            actor_critic.train()
            if solve_score is not None:
                solved = all(r >= solve_score for (t, r) in episode_info)

        # Save model
        if (epoch % save_every == 0) or (epoch == epochs) or solved:
            logger.save_state({'env': env}, None)

        # Check environment is solved
        if solved:
            plog = lambda msg: logger.log(msg, color='green')
            plog("=" * 40)
            plog(f"ENVIRONMENT SOLVED!")
            plog("=" * 40)
            plog(f'    TotalEnvInteracts {total_steps}')
            plog(f'    Time {time.time() - start_time}')
            plog(f'    Epoch {epoch}')
            break


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--env', type=str, default='CartPole-v1')
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--allow_run_as_root', action='store_true')
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--num_hidden', type=int, default=2)
    parser.add_argument('--gamma', '-g', type=float, default=0.99)
    parser.add_argument('--tau', '-t', type=float, default=0.99)
    parser.add_argument('--max_grad_norm', type=float, default=None)
    parser.add_argument('--use_gae', type=bool, default=True)
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
        from spinup.algos.a2c import core

        model = core.make_model(
            env,
            model_kwargs=dict(hidden_sizes=[args.hidden_size] * args.num_hidden),
        )

    # Setup epoch number and action frequencies
    epochs = args.epochs // args.cpu
    save_every = args.save_every or max(10, epochs // 10)
    log_every = args.log_every or max(10, epochs // 10)
    test_every = args.test_every or max(10, epochs // 10)

    assert save_every <= epochs
    assert log_every <= epochs
    assert test_every <= epochs

    if args.cpu > 1:
        mpi_tools.mpi_fork(args.cpu, allow_run_as_root=args.allow_run_as_root)  # run parallel code with mpi

    train(
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
        solve_score=args.solve_score,
    )
