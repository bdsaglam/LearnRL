import numpy as np


def calculate_returns(rewards: np.ndarray,
                      dones: np.ndarray,
                      next_value: [np.ndarray, float],
                      discount_factor: float) -> np.ndarray:
    if rewards.shape != dones.shape:
        raise ValueError("rewards and dones must have same shape; either (steps, ) or (batch_size, steps)")
    if rewards.ndim == 1:
        if not isinstance(next_value, float) or (isinstance(next_value, np.ndarray) and next_value.ndim != 0):
            raise ValueError(f"next_value must be a float scalar")
    if rewards.ndim == 2:
        batch_size, steps = rewards.shape
        if next_value.shape != (batch_size, 1):
            raise ValueError(f"next_value's shape must be ({batch_size}, 1)")

    # Bellman backup for Q function
    # Q(s_t,a_t) = R_t + gamma * V(s_t+1)
    returns = np.zeros_like(rewards)
    for i in reversed(range(returns.size)):
        returns[i] = rewards[i] + discount_factor * (1 - dones[i]) * next_value
        next_value = returns[i]
    return returns
