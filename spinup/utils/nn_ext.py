import torch.nn as nn


def MLP(sizes, activation=nn.ReLU, activate_final=True, bias=True):
    n = len(sizes)
    assert n >= 2, "There must be at least two sizes"

    layers = []
    for j in range(n - 1):
        layers.append(nn.Linear(sizes[j], sizes[j + 1], bias=bias))
        layers.append(activation())

    if not activate_final:
        layers.pop()

    return nn.Sequential(*layers)


def Conv2dStack(in_channels,
                out_channels,
                kernel_sizes,
                strides,
                activation=nn.ReLU,
                activate_final=True):
    assert len(out_channels) == len(kernel_sizes) == len(strides)

    channels = [in_channels] + list(out_channels)
    layers = []
    for i in range(len(channels) - 1):
        in_channels = channels[i]
        out_channels = channels[i + 1]
        kernel_size = kernel_sizes[i]
        stride = strides[i]
        conv = nn.Conv2d(in_channels=in_channels,
                         out_channels=out_channels,
                         kernel_size=kernel_size,
                         stride=stride)
        layers.append(conv)
        layers.append(activation())

    if not activate_final:
        layers.pop()

    return nn.Sequential(*layers)
