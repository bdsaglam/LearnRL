import torch
import torch.nn as nn

from spinup.algos.a2c.core import ActorCritic
from spinup.core.approximators import MLPCategoricalActor, MLPVFunction
from spinup.utils import nn_utils


def make_feature_extractor(input_shape):
    c, h, w = input_shape

    network = nn.Sequential(
        nn.Conv2d(c, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(64, 128, kernel_size=7, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Flatten()
    )

    with torch.no_grad():
        feature_dim = network(torch.rand(1, c, h, w)).shape

    return network, feature_dim[-1]


def make_model(env, model_kwargs):
    act_dim = env.action_space.n

    # Create actor-critic module and target networks
    feature_extractor, feature_dim = make_feature_extractor(env.observation_space.shape)
    actor_network = MLPCategoricalActor(feature_dim, act_dim, **model_kwargs)
    critic_network = MLPVFunction(feature_dim, **model_kwargs)
    ac = ActorCritic(
        feature_extractor=feature_extractor,
        actor=actor_network,
        critic=critic_network
    )

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(nn_utils.count_vars(module) for module in [actor_network, critic_network])
    print('\nNumber of parameters: \t actor: %d, \t critic: %d\n' % var_counts)

    return ac
