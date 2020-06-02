import numpy as np

import torch.nn as nn


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def conv_output_size(in_size, kernel, stride=1, padding=0):
    return (in_size - kernel + 2 * padding) // stride + 1


def trainable(module, requires_grad):
    for p in module.parameters():
        p.requires_grad = requires_grad
