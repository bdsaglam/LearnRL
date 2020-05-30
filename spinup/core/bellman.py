import numpy as np
import scipy.signal


def discounted_reward_sum(rewards, discount_factor):
    """
    Calculates discounted cumulative reward.

    :param rewards: array of reward value for each time step
    :param discount_factor: discount factor [0, 1]
    :return: total discounted reward for each time step
    [r0 + discount_factor * r1 + discount_factor^2 * r2, r1 + discount_factor * r2, r2]
    """
    result = scipy.signal.lfilter([1], [1, float(-discount_factor)], rewards[::-1], axis=0)[::-1]
    return np.copy(result)


def generalized_advantage_estimate(rewards: np.ndarray,
                                   values: np.ndarray,
                                   next_value: float,
                                   discount_factor: float,
                                   tau: float,
                                   ) -> np.ndarray:
    values = np.append(values, next_value)
    deltas = rewards + discount_factor * values[1:] - values[:-1]
    advantages = discounted_reward_sum(deltas, discount_factor * tau)
    return advantages


def calculate_returns(rewards: np.ndarray,
                      next_value: float,
                      discount_factor: float,
                      ) -> np.ndarray:
    # Bellman backup for Q function
    # Q(s_t,a_t) = R_t + gamma * V(s_t+1)
    x = np.append(rewards, next_value)
    return np.copy(discounted_reward_sum(x, discount_factor)[:-1])


def calculate_batch_returns(rewards: np.ndarray,
                            dones: np.ndarray,
                            next_value: np.ndarray,
                            discount_factor: float) -> np.ndarray:
    if rewards.shape != dones.shape:
        raise ValueError("rewards and dones must have same shape; (batch_size, num_steps)")

    batch_size, num_steps = rewards.shape
    if next_value.shape != (batch_size, 1):
        raise ValueError(f"next_value's shape must be ({batch_size}, 1)")

    # Bellman backup for Q function
    # Q(s_t,a_t) = R_t + gamma * V(s_t+1)
    # TODO: this seems wrong
    # slices must be [:, i]
    returns = np.zeros_like(rewards)
    for i in reversed(range(num_steps)):
        returns[i] = rewards[i] + discount_factor * (1 - dones[i]) * next_value
        next_value = returns[i]
    return returns
