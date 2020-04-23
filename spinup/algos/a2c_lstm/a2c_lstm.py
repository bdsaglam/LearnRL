import itertools
import time
from copy import deepcopy

import gym
import numpy as np
import torch
from torch.optim import Adam

import spinup.algos.a2c_lstm.core as core
import spinup.algos.a2c_lstm.utils as utils
from spinup.utils.logx import EpochLogger


def compute_returns(rewards, dones, next_value, gamma):
    returns = np.zeros_like(rewards)
    for i in reversed(range(returns.size)):
        returns[i] = rewards[i] + gamma * (1 - dones[i]) * next_value
        next_value = returns[i]
    return returns


def compute_loss(ac, features, log_probs, entropies, returns, entropy_reg_coef):
    v1 = ac.v1(features)
    v2 = ac.v2(features)

    # MSE loss against Bellman backup
    loss_v1 = (returns - v1).pow(2).mean()
    loss_v2 = (returns - v2).pow(2).mean()
    loss_v = loss_v1 + loss_v2

    # Useful info for logging
    v_info = dict(V1Vals=v1.detach().numpy(),
                  V2Vals=v2.detach().numpy())

    values = torch.min(v1.detach(), v2.detach())
    advantages = returns - values

    # Entropy-regularized policy loss
    loss_pi = 1 * (-entropy_reg_coef * entropies.mean() - (advantages * log_probs).mean())

    # Useful info for logging
    pi_info = dict(LogPi=log_probs.detach().numpy(), MeanEntropy=entropies.detach().mean().numpy())

    return loss_v, v_info, loss_pi, pi_info


def a2c_lstm(env_fn,
             actor_critic=core.ActorCritic,
             actor_kwargs=dict(),
             critic_kwargs=dict(),
             seed=0,
             steps_per_epoch=1000,
             epochs=1000,
             gamma=0.99,
             polyak=0.995,
             lr=1e-3,
             entropy_reg_coef=0.2,
             logger_kwargs=dict(),
             log_every_epoch=100,
             save_freq=1,
             num_test_episodes=10,
             test_episode_len_limit=1000,
             ):
    """
    Advantage Actor-Critic (A2C)


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with an ``act`` 
            method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
            The ``act`` method and ``pi`` module should accept batches of 
            observations as inputs, and ``q1`` and ``q2`` should accept a batch 
            of observations and a batch of actions as inputs. When called, 
            ``act``, ``q1``, and ``q2`` should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``q1``       (batch,)          | Tensor containing one current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ``q2``       (batch,)          | Tensor containing the other current 
                                           | estimate of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================

            Calling ``pi`` should return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``logp_pi``  (batch,)          | Tensor containing log probabilities of
                                           | actions in ``a``. Importantly: gradients
                                           | should be able to flow back into ``a``.
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to SAC.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target 
            networks. Target networks are updated towards main networks 
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow 
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
            close to 1.)

        lr (float): Learning rate (used for both policy and value learning).

        alpha (float): Entropy regularization coefficient. (Equivalent to 
            inverse of reward scale in the original SAC paper.)

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long 
            you wait between updates, the ratio of env steps to gradient steps 
            is locked to 1.

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    torch.manual_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    feature_dim = 64

    # Create actor-critic module and target networks
    feature_extractor = core.FeatureExtractor(obs_dim, feature_dim)
    actor_network = core.MLPCategoricalActor(feature_dim, act_dim, **actor_kwargs)
    critic_network = core.MLPVFunction(feature_dim, **critic_kwargs)
    ac = actor_critic(feature_extractor=feature_extractor, actor_network=actor_network, critic_network=critic_network)
    ac_targ = deepcopy(ac)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False

    # List of parameters for both Q-networks (save this for convenience)
    v_params = itertools.chain(ac.v1.parameters(), ac.v2.parameters())

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(utils.count_vars(module) for module in [ac.fe, ac.pi, ac.v1, ac.v2])
    logger.log('\nNumber of parameters: \t fe: %d, \t pi: %d, \t v1: %d, \t v2: %d\n' % var_counts)

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=lr)
    v_optimizer = Adam(v_params, lr=lr)
    fe_optimizer = Adam(ac.fe.parameters(), lr=lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update(features, log_probs, entropies, returns):
        # First run one gradient descent step for Q1 and Q2
        v_optimizer.zero_grad()
        pi_optimizer.zero_grad()
        loss_v, v_info, loss_pi, pi_info = compute_loss(ac,
                                                        features=features,
                                                        log_probs=log_probs,
                                                        entropies=entropies,
                                                        returns=returns,
                                                        entropy_reg_coef=entropy_reg_coef)
        loss = loss_v + loss_pi
        loss.backward()
        v_optimizer.step()
        pi_optimizer.step()
        fe_optimizer.step()

        # Record things
        logger.store(LossV=loss_v.item(), **v_info)
        logger.store(LossPi=loss_pi.item(), **pi_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o):
        return ac.act(torch.as_tensor(o, dtype=torch.float32))

    def test_agent():
        agent = core.TestAgent(ac)
        for j in range(num_test_episodes):
            agent.reset()
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not (d or (ep_len == test_episode_len_limit)):
                # Take deterministic actions at test time
                with torch.no_grad():
                    obs_tensor = torch.as_tensor(o, dtype=torch.float32)
                    a = agent.act_deterministic(obs_tensor)
                o, r, d, _ = test_env.step(a)
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    # Prepare for interaction with environment
    start_time = time.time()
    # Main loop: collect experience in env and update/log each epoch
    total_steps = 0

    # Reset env
    obs = env.reset()
    hx, cx = ac.fe.initial_hidden_state()
    # Reset episode stats
    ep_ret = 0
    ep_len = 0
    for epoch in range(epochs):
        observation_tensors = []
        feature_tensors = []
        rewards = []
        dones = []
        log_prob_tensors = []
        entropy_tensors = []
        for t in range(steps_per_epoch):
            total_steps += 1

            obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
            hx, cx = ac.fe(obs_tensor.unsqueeze(0), (hx, cx))
            feature_tensor = hx.squeeze(0)
            dist = ac.pi(feature_tensor)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

            # Step the env
            obs2, reward, done, _ = env.step(action.detach().numpy())
            ep_ret += reward
            ep_len += 1

            # Store experience to buffer
            observation_tensors.append(obs_tensor)
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
                logger.store(EpRet=ep_ret, EpLen=ep_len)
                # Reset env
                obs = env.reset()
                hx, cx = ac.fe.initial_hidden_state()
                # Reset episode stats
                ep_ret = 0
                ep_len = 0
                break

        # Update
        if len(rewards) > 0:
            if dones[-1]:
                next_value = 0
            else:
                # Bellman backup for Q functions
                with torch.no_grad():
                    obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
                    hx, cx = ac.fe(obs_tensor.unsqueeze(0), (hx, cx))
                    feature_tensor = hx.squeeze(0)
                    next_value1 = ac_targ.v1(feature_tensor)
                    next_value2 = ac_targ.v2(feature_tensor)
                    next_value = min(next_value1.item(), next_value2.item())

            batch_observation = torch.stack(observation_tensors)
            batch_feature = torch.stack(feature_tensors)
            batch_log_prob = torch.stack(log_prob_tensors).unsqueeze(-1)
            batch_entropy = torch.stack(entropy_tensors).unsqueeze(-1)
            returns = compute_returns(rewards=rewards, dones=dones, next_value=next_value, gamma=gamma)
            batch_return = torch.tensor(returns, dtype=torch.float32).unsqueeze(-1)

            update(batch_feature, batch_log_prob, batch_entropy, batch_return)

        # End of epoch handling
        if epoch % log_every_epoch == 0:
            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs):
                logger.save_state({'env': env}, None)

            # Test the performance of the deterministic version of the agent.
            test_agent()

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', total_steps)
            logger.log_tabular('V1Vals', with_min_and_max=True)
            logger.log_tabular('V2Vals', with_min_and_max=True)
            logger.log_tabular('LogPi', with_min_and_max=True)
            logger.log_tabular('MeanEntropy', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossV', average_only=True)
            logger.log_tabular('Time', time.time() - start_time)
            logger.dump_tabular()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='LunarLander-v2')
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--num_hidden', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--steps_per_epoch', type=int, default=1000)
    parser.add_argument('--exp_name', type=str, default='a2c_lstm')
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    torch.set_num_threads(torch.get_num_threads())

    a2c_lstm(lambda: gym.make(args.env),
             actor_critic=core.ActorCritic,
             actor_kwargs=dict(hidden_sizes=[args.hidden_size] * args.num_hidden),
             critic_kwargs=dict(hidden_sizes=[args.hidden_size] * args.num_hidden),
             gamma=args.gamma,
             seed=args.seed,
             epochs=args.epochs,
             steps_per_epoch=args.steps_per_epoch,
             logger_kwargs=logger_kwargs)
