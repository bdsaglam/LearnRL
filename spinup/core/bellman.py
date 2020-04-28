import numpy as np
import scipy.signal


def calculate_returns(rewards: np.ndarray,
                      dones: np.ndarray,
                      next_value: [np.ndarray, float],
                      discount_factor: float) -> np.ndarray:
    if rewards.shape != dones.shape:
        raise ValueError("rewards and dones must have same shape; either (steps, ) or (batch_size, steps)")
    if rewards.ndim == 1:
        if not isinstance(next_value, (int, float)) or (isinstance(next_value, np.ndarray) and next_value.ndim != 0):
            raise ValueError(f"next_value must be a float scalar")
        if isinstance(next_value, int):
            next_value = float(next_value)
    if rewards.ndim == 2:
        batch_size, steps = rewards.shape
        if next_value.shape != (batch_size, 1):
            raise ValueError(f"next_value's shape must be ({batch_size}, 1)")

    # Bellman backup for Q function
    # Q(s_t,a_t) = R_t + gamma * V(s_t+1)
    num_steps = rewards.shape[-1]
    returns = np.zeros_like(rewards)
    for i in reversed(range(num_steps)):
        returns[i] = rewards[i] + discount_factor * (1 - dones[i]) * next_value
        next_value = returns[i]
    return returns


def discount_cumsum(x, discount_factor):
    """
    Calculates discounted cumulative sum of an array.

    input:
        array x,
        [x0, x1, x2]

    output:
        [x0 + discount_factor * x1 + discount_factor^2 * x2, x1 + discount_factor * x2, x2]
    """
    result = scipy.signal.lfilter([1], [1, float(-discount_factor)], x[::-1], axis=0)[::-1]
    return np.copy(result)


def generalized_advantage_estimate(rewards: np.ndarray,
                                   values: np.ndarray,
                                   next_value: float,
                                   discount_factor: float,
                                   tau: float,
                                   ):
    values = np.append(values, next_value)
    deltas = rewards + discount_factor * values[1:] - values[:-1]
    advantages = discount_cumsum(deltas, discount_factor * tau)
    return advantages
