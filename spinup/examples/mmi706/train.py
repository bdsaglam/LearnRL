import pathlib
import time
from copy import deepcopy

import numpy as np
import torch
from gym.wrappers import Monitor
from torch.optim import Adam

from spinup.constants import DEVICE
from spinup.examples.mmi706.agent import make_agent, Agent
from spinup.examples.mmi706.environment import make_env
from spinup.examples.mmi706.evaluate import evaluate_agent
from spinup.utils import mpi_tools, mpi_pytorch
from spinup.utils.experiment_utils import get_latest_saved_file
from spinup.utils.logx import EpochLogger
from spinup.utils.storage import EpisodeHistory


def a2c(env_fn,
        agent: Agent,
        seed=0,
        num_cpu=1,
        device=torch.device("cpu"),
        epochs=1000,
        steps_per_epoch=100,
        gamma=0.99,
        use_gae=True,
        tau=0.95,
        max_grad_norm=0.5,
        polyak=0.995,
        learning_rate=1e-3,
        value_loss_coef=0.5,
        policy_loss_coef=1,
        entropy_loss_coef=0.1,
        grid_layer_weight_reg_loss_coef=1e-4,
        save_every=100,
        log_every=10,
        logger_kwargs=dict(),
        test_every=100,
        num_test_episodes=5,
        deterministic=False,
        save_freq=1,
        solved_score=None,
        render=False,
        ):
    use_MPI = num_cpu > 1

    if use_MPI:
        # Special function to avoid certain slowdowns from PyTorch + MPI combo.
        mpi_pytorch.setup_pytorch_for_mpi()
    else:
        torch.set_num_threads(torch.get_num_threads())

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    config = locals()
    del config['env_fn']
    del config['agent']
    del config['logger']
    logger.save_config(config)

    test_logger_kwargs = deepcopy(logger_kwargs)
    test_logger_kwargs['output_dir'] = pathlib.Path(test_logger_kwargs['output_dir']) / 'evaluation'
    test_logger = EpochLogger(**test_logger_kwargs)

    # Random seed
    if use_MPI:
        seed += 10000 * mpi_tools.proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    env = env_fn()

    assert env.max_episode_steps > 0

    obs_shape = env.observation_space.shape
    act_dim = env.action_space.n

    # training model and target model
    target_agent = deepcopy(agent)
    if use_MPI:
        # Sync params across processes
        mpi_pytorch.sync_params(agent)
        mpi_pytorch.sync_params(target_agent)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in target_agent.parameters():
        p.requires_grad = False

    # Utilize GPU
    agent.to(device)
    target_agent.to(device)

    # Set up optimizers for policy and q-function
    optimizer = Adam(agent.parameters(), lr=learning_rate)

    # Set up model saving
    logger.setup_pytorch_saver(agent, name='model')

    def update(episode_buffer):
        # Update
        if episode_buffer.dones[-1]:
            next_value = 0.0
        else:
            last_obs = episode_buffer.next_observations[-1]
            previous_reward = episode_buffer.rewards[-1]
            last_obs_tensor = torch.tensor(last_obs, dtype=torch.float32).unsqueeze(0)
            previous_reward_tensor = torch.tensor([previous_reward], dtype=torch.float32).unsqueeze(0)
            context = agent.get_context()
            next_value = target_agent.predict_value(obs_tensor=last_obs_tensor,
                                                    previous_reward_tensor=previous_reward_tensor,
                                                    goal_grid_code_tensor=goal_grid_code_tensor,
                                                    context=context).cpu().item()

        # Super critical!!
        optimizer.zero_grad()

        # Compute value and policy losses
        loss, info = agent.compute_loss(rewards=np.array(episode_buffer.rewards),
                                        dones=np.array(episode_buffer.dones),
                                        next_value=next_value,
                                        discount_factor=gamma,
                                        use_gae=use_gae,
                                        tau=tau,
                                        value_loss_coef=value_loss_coef,
                                        policy_loss_coef=policy_loss_coef,
                                        entropy_reg_coef=entropy_loss_coef,
                                        grid_layer_wreg_loss_coef=grid_layer_weight_reg_loss_coef)
        loss.backward()
        if use_MPI:
            mpi_pytorch.mpi_avg_grads(agent)

        # Optimize
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
        optimizer.step()

        # Log losses and info
        logger.store(**info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(agent.parameters(), target_agent.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)
        if use_MPI:
            mpi_pytorch.sync_params(target_agent)

    # Prepare for interaction with environment
    start_time = time.time()

    # Main loop: collect experience in env and update/log each epoch
    total_steps = 0

    # Reset env
    obs = env.reset()
    reward = 0
    goal_grid_code_tensor = None

    # Reset episode stats
    episode_return = 0
    episode_length = 0

    for epoch in range(1, epochs + 1):
        agent.reset_for_training()
        epoch_history = EpisodeHistory()
        for t in range(steps_per_epoch):
            total_steps += 1

            # Get action from the model
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            previous_reward_tensor = torch.tensor([reward], dtype=torch.float32).unsqueeze(0)
            action = agent.step(obs_tensor, previous_reward_tensor, goal_grid_code_tensor).squeeze(0)

            # Step the env
            obs2, reward, done, _ = env.step(action.detach().cpu().item())
            if render and mpi_tools.proc_id() == 0:
                env.render('human', view='top')
                time.sleep(1e-3)
            episode_return += reward
            episode_length += 1

            # Store transition to history
            epoch_history.store(observation=None, action=None, reward=reward, done=done, next_observation=obs2)

            # Super critical, easy to overlook step: make sure to update
            # most recent observation!
            obs = obs2

            # End of trajectory handling
            if done:
                if reward > 0:
                    goal_grid_code_tensor = agent.current_grid_code.detach()
                break

        update(epoch_history)

        # if done
        if epoch_history.dones[-1]:
            logger.store(EpRet=episode_return, EpLen=episode_length)
            # Reset env
            obs = env.reset()
            agent.reset()
            # Reset episode stats
            episode_return = 0
            episode_length = 0

        # End of epoch handling
        if epoch % log_every == 0:
            total_interactions = mpi_tools.mpi_sum(total_steps) if use_MPI else total_steps

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('Value', average_only=True)
            logger.log_tabular('LogPi', with_min_and_max=True)
            logger.log_tabular('LossV', average_only=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossEntropy', average_only=True)
            logger.log_tabular('LossGridL2', average_only=True)
            logger.log_tabular('LossRecons', average_only=True)
            logger.log_tabular('TotalEnvInteracts', total_interactions)
            logger.log_tabular('Time', time.time() - start_time)
            logger.dump_tabular()

        # Test agent
        solved = False
        if epoch % test_every == 0:
            video_dir = pathlib.Path(logger.output_dir) / 'test_videos' / f'epoch-{epoch:d}'
            test_env_fn = lambda: Monitor(env_fn(), directory=video_dir)
            # Test the performance of the deterministic version of the agent.
            context = agent.get_context()
            agent.eval()
            episode_info = evaluate_agent(env_fn=test_env_fn,
                                          agent=agent,
                                          deterministic=deterministic,
                                          num_episodes=num_test_episodes,
                                          render=False,
                                          logger=test_logger)
            agent.train()
            agent.set_context(context)
            if solved_score is not None:
                solved = all(r >= solved_score for (t, r) in episode_info)

        # Save model
        if (epoch % save_every == 0) or (epoch == epochs) or solved:
            logger.save_state({'env': env})

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

    env.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--allow_run_as_root', action='store_true')
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--steps_per_epoch', type=int, default=100)
    parser.add_argument('--max_episode_steps', type=int, default=1500)
    parser.add_argument('--gamma', '-g', type=float, default=0.99)
    parser.add_argument('--tau', '-t', type=float, default=0.99)
    parser.add_argument('--max_grad_norm', type=float, default=0.5)
    parser.add_argument('--use_gae', type=bool, default=True)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--value_loss_coef', '-vl', type=float, default=0.5)
    parser.add_argument('--policy_loss_coef', '-pl', type=float, default=1)
    parser.add_argument('--entropy_loss_coef', '-el', type=float, default=0.1)
    parser.add_argument('--grid_layer_weight_reg_loss_coef', '-gl', type=float, default=1e-4)
    parser.add_argument('--save_every', type=int, default=None)
    parser.add_argument('--log_every', type=int, default=None)
    parser.add_argument('--test_every', type=int, default=None)
    parser.add_argument('--num_test_episodes', type=int, default=5)
    parser.add_argument('--deterministic', '-d', action='store_true')
    parser.add_argument('--solved_score', type=int, default=None)
    parser.add_argument('--continue_training', '-c', action='store_true')
    parser.add_argument('--saved_model_file', '-f', type=str, default=None)
    parser.add_argument('--render', action='store_true')

    args = parser.parse_args()

    if args.cpu > 1:
        mpi_tools.mpi_fork(args.cpu, allow_run_as_root=args.allow_run_as_root)  # run parallel code with mpi

    # Setup experiment name
    env = make_env(args.seed)

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
        model = make_agent(
            env,
            pim_lstm_hidden_size=32,
            grid_layer_size=32,
            grid_layer_dropout_rate=0.5,
            ac_lstm_hidden_size=32,
            actor_hidden_sizes=(32,),
            critic_hidden_sizes=(32,)
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
        env_fn=lambda: make_env(args.seed, args.max_episode_steps),
        agent=model,
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
        grid_layer_weight_reg_loss_coef=args.grid_layer_weight_reg_loss_coef,
        epochs=epochs,
        steps_per_epoch=args.steps_per_epoch,
        save_every=save_every,
        log_every=log_every,
        logger_kwargs=logger_kwargs,
        test_every=test_every,
        num_test_episodes=args.num_test_episodes,
        deterministic=args.deterministic,
        solved_score=args.solved_score,
        render=args.render,
    )
