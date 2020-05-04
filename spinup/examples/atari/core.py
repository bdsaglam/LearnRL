import torch
import torch.nn as nn

from spinup.core.approximators import MLPCategoricalActor, MLPVFunction
from spinup.examples.classic_control.core import ActorDoubleCritic
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
    ac = ActorDoubleCritic(
        feature_extractor=feature_extractor,
        actor=actor_network,
        critic=critic_network
    )

    # Count variables (pro-tip: try to get a feel for how different size networks behave!)
    print('\nNumber of parameters')
    print('-' * 32)
    print(f'Feature extractor: \t {nn_utils.count_vars(feature_extractor):d}')
    print(f'Actor network: \t {nn_utils.count_vars(actor_network):d}')
    print(f'Critic network: \t {nn_utils.count_vars(critic_network):d}')
    print(f'Agent: \t {nn_utils.count_vars(ac):d}')

    return ac
