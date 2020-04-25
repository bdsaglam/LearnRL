import torch
import torch.nn as nn

from spinup.algos.a2c.core import ActorCritic
from spinup.core.approximators import MLPCategoricalActor, MLPVFunction
from spinup.examples.atari.feature_extraction import frames_feature_extractor, make_frame_buffer, preprocess
from spinup.utils import nn_utils


class AtariActorCritic(ActorCritic):
    def __init__(self, feature_extractor: nn.Module, actor: nn.Module, critic: nn.Module):
        super().__init__(feature_extractor=feature_extractor, actor=actor, critic=critic)
        self.frame_buffer = make_frame_buffer()

    def reset(self):
        super().reset()
        self.frame_buffer = make_frame_buffer()

    def step(self, obs_tensor):
        device = obs_tensor.device
        frame_tensor = preprocess(obs_tensor.cpu().numpy()).to(device)
        self.frame_buffer.append(frame_tensor)
        frames = list(self.frame_buffer)
        multi_frame_tensor = torch.cat(frames, dim=0)
        return super().step(multi_frame_tensor)

    def predict_value(self, obs_tensor):
        device = obs_tensor.device
        frame_tensor = preprocess(obs_tensor.cpu().numpy()).to(device)
        frames = list(self.frame_buffer)[1:] + [frame_tensor]
        multi_frame_tensor = torch.cat(frames, dim=0)
        return super().predict_value(multi_frame_tensor)

    def act(self, obs, deterministic=False):
        frame_tensor = preprocess(obs)
        self.frame_buffer.append(frame_tensor)
        frames = list(self.frame_buffer)
        multi_frame_tensor = torch.cat(frames, dim=0)
        return super().act(multi_frame_tensor.numpy(), deterministic)


def make_model(env, model_kwargs):
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    # Create actor-critic module and target networks
    feature_extractor, feature_dim = frames_feature_extractor()
    actor_network = MLPCategoricalActor(feature_dim, act_dim, **model_kwargs)
    critic_network = MLPVFunction(feature_dim, **model_kwargs)
    ac = AtariActorCritic(
        feature_extractor=feature_extractor,
        actor=actor_network,
        critic=critic_network
    )

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(nn_utils.count_vars(module) for module in [actor_network, critic_network])
    print('\nNumber of parameters: \t actor: %d, \t critic: %d\n' % var_counts)

    return ac
