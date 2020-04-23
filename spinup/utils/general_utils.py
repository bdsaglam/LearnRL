import pathlib
import re

import numpy as np
import scipy.signal


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def get_latest_file_iteration(folder, pattern='*'):
    folder = pathlib.Path(folder)
    matches = [(fp, re.findall(r'\d+', fp.stem)) for fp in folder.glob(pattern)]
    file_itr_pairs = [(fp, int(m[-1])) for fp, m in matches if len(m) > 0]
    if len(file_itr_pairs) == 0:
        return None, None
    return max(file_itr_pairs, key=lambda t: t[1])
