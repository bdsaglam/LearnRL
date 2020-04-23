import pathlib
import time
from copy import deepcopy

import gym
import numpy as np
import torch
from torch.optim import Adam

from spinup.algos.a2c import core
from spinup.utils import nn_utils
from spinup.utils.evaluate import evaluate_agent
from spinup.utils.experiment_utils import get_latest_saved_file
from spinup.utils.logx import EpochLogger


def compute_returns(rewards, dones, next_value, gamma):
    # Bellman backup for Q function
    # Q(s_t,a_t) = R_t + gamma * V(s_t+1)
    returns = np.zeros_like(rewards)
    for i in reversed(range(returns.size)):
        returns[i] = rewards[i] + gamma * (1 - dones[i]) * next_value
        next_value = returns[i]
    return returns


def make_model(env, model_kwargs):
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    # Create actor-critic module and target networks
    feature_dim = obs_dim
    actor_network = core.MLPCategoricalActor(feature_dim, act_dim, **model_kwargs)
    critic_network = core.MLPVFunction(feature_dim, **model_kwargs)
    ac = core.ActorCritic(
        actor=actor_network,
        critic=critic_network)

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(nn_utils.count_vars(module) for module in [actor_network, critic_network])
    print('\nNumber of parameters: \t actor: %d, \t critic: %d\n' % var_counts)

    return ac


def train(env,
          test_env,
          model,
          seed=0,
          steps_per_epoch=10,
          epochs=1000,
          gamma=0.99,
          polyak=0.995,
          lr=1e-3,
          entropy_reg_coef=0.2,
          save_every=100,
          log_every=100,
          logger_kwargs=dict(),
          test_every=100,
          num_test_episodes=5,
          test_episode_len_limit=None,
          save_freq=1,
          device=torch.device("cpu"),
          ):
    logger = EpochLogger(**logger_kwargs)
    config = locals()
    del config['env']
    del config['test_env']
    del config['model']
    del config['logger']
    config['env_name'] = env.spec.id
    logger.save_config(config)

    torch.manual_seed(seed)
    np.random.seed(seed)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    ac = model
    ac_targ = deepcopy(ac)

    test_agent = core.Agent(ac.actor, device)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.actor_parameters(), lr=lr)
    v_optimizer = Adam(ac.critic_parameters(), lr=lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac, name='model')
    logger.setup_pytorch_saver(test_agent, name='agent')

    def update(features, log_probs, entropies, returns):
        # Super critical!
        v_optimizer.zero_grad()
        pi_optimizer.zero_grad()
        # Compute value and policy losses
        loss_v, v_info, loss_pi, pi_info = ac.compute_loss(features=features,
                                                           log_probs=log_probs,
                                                           entropies=entropies,
                                                           returns=returns,
                                                           entropy_reg_coef=entropy_reg_coef)
        loss = loss_v + loss_pi
        loss.backward()

        # Optimize
        v_optimizer.step()
        pi_optimizer.step()

        # Record things
        logger.store(LossV=loss_v.cpu().detach().item(), **v_info)
        logger.store(LossPi=loss_pi.cpu().detach().item(), **pi_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    # Utilize GPU
    ac.to(device)
    ac_targ.to(device)
    # Prepare for interaction with environment
    start_time = time.time()
    # Main loop: collect experience in env and update/log each epoch
    total_steps = 0
    # Reset env
    obs = env.reset()
    # Reset episode stats
    episode_return = 0
    episode_length = 0
    logger.store(EpRet=0, EpLen=0)
    for epoch in range(epochs):
        feature_tensors = []
        rewards = []
        dones = []
        log_prob_tensors = []
        entropy_tensors = []
        for t in range(steps_per_epoch):
            total_steps += 1

            feature_tensor = torch.tensor(obs, dtype=torch.float32)
            dist = ac.infer_action_dist(feature_tensor)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

            # Step the env
            obs2, reward, done, _ = env.step(action.detach().cpu().numpy())
            episode_return += reward
            episode_length += 1

            # Store experience to buffer
            feature_tensors.append(feature_tensor)
            log_prob_tensors.append(log_prob)
            entropy_tensors.append(entropy)
            rewards.append(reward)
            dones.append(done)

            # Super critical, easy to overlook step: make sure to update
            # most recent observation!
            obs = obs2

            # End of trajectory handling
            if done:
                logger.store(EpRet=episode_return, EpLen=episode_length)
                # Reset env
                obs = env.reset()
                # Reset episode stats
                episode_return = 0
                episode_length = 0
                break

        # Update
        if dones[-1]:
            next_value = 0
        else:
            # Bellman backup for Q function
            # Q(s_t,a_t) = R_t + gamma * V(s_t+1)
            with torch.no_grad():
                feature_tensor = torch.tensor(obs, dtype=torch.float32)
                next_value = ac_targ.infer_value(feature_tensor).cpu().item()

        batch_feature = torch.stack(feature_tensors).to(device)
        batch_log_prob = torch.stack(log_prob_tensors).unsqueeze(-1).to(device)
        batch_entropy = torch.stack(entropy_tensors).unsqueeze(-1).to(device)
        returns = compute_returns(rewards=rewards,
                                  dones=dones,
                                  next_value=next_value,
                                  gamma=gamma)
        batch_return = torch.tensor(returns, dtype=torch.float32).unsqueeze(-1).to(device)

        update(batch_feature, batch_log_prob, batch_entropy, batch_return)

        # End of epoch handling
        if epoch % log_every == 0:
            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('V1Vals', with_min_and_max=True)
            logger.log_tabular('V2Vals', with_min_and_max=True)
            logger.log_tabular('LogPi', with_min_and_max=True)
            logger.log_tabular('MeanEntropy', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossV', average_only=True)
            logger.log_tabular('TotalEnvInteracts', total_steps)
            logger.log_tabular('Time', time.time() - start_time)
            logger.dump_tabular()

        if epoch % test_every == 0:
            # Test the performance of the deterministic version of the agent.
            evaluate_agent(env=test_env,
                           agent=test_agent,
                           deterministic=False,
                           num_episodes=num_test_episodes,
                           episode_len_limit=test_episode_len_limit,
                           render=False)
        # Save model
        if (epoch % save_every == 0) or (epoch == epochs):
            logger.save_state({'env': env}, None)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--env', type=str, default='CartPole-v1')
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--num_hidden', type=int, default=2)
    parser.add_argument('--gamma', '-g', type=float, default=0.99)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--steps_per_epoch', type=int, default=10)
    parser.add_argument('--continue_training', '-c', action='store_true')
    parser.add_argument('--saved_model_file', '-f', type=str, default=None)
    parser.add_argument('--save_every', type=int, default=100)
    parser.add_argument('--log_every', type=int, default=100)
    parser.add_argument('--test_every', type=int, default=100)
    parser.add_argument('--num_test_episodes', type=int, default=5)
    parser.add_argument('--test_episode_len_limit', type=int, default=None)

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
        assert save_dir.exists()
        saved_model_file = get_latest_saved_file(save_dir, prefix='model')

    if saved_model_file:
        assert saved_model_file.exists()
        model = torch.load(saved_model_file)
        for p in model.parameters():
            p.requires_grad_()
        print("Loaded model from: ", saved_model_file)
    else:
        model = make_model(
            env,
            model_kwargs=dict(hidden_sizes=[args.hidden_size] * args.num_hidden),
        )
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    train(
        env=env,
        test_env=gym.make(args.env),
        model=model,
        gamma=args.gamma,
        seed=args.seed,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        save_every=args.save_every,
        log_every=args.log_every,
        logger_kwargs=logger_kwargs,
        test_every=args.test_every,
        num_test_episodes=args.num_test_episodes,
        test_episode_len_limit=args.test_episode_len_limit,
        device=device,
    )
