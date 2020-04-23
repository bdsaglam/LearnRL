import numpy as np


def calculate_returns(rewards, dones, next_value, discount_factor):
    # Bellman backup for Q function
    # Q(s_t,a_t) = R_t + gamma * V(s_t+1)
    returns = np.zeros_like(rewards)
    for i in reversed(range(returns.size)):
        returns[i] = rewards[i] + discount_factor * (1 - dones[i]) * next_value
        next_value = returns[i]
    return returns
