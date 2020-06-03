from typing import Any

import torch
import torch.nn.functional as F
from spinup.constants import DEVICE
from torch import nn

from spinup.core.bellman import calculate_returns, \
    generalized_advantage_estimate
from spinup.examples.mmi706.actor_critic import ActorCriticModule
from spinup.examples.mmi706.path_integration import PathIntegrationModule
from spinup.examples.mmi706.vision import make_vision_module
from spinup.utils import nn_utils
from spinup.utils.nn_utils import trainable


class TrainBuffer:
    def __init__(self):
        self.log_probs = []
        self.entropy = []
        self.v1 = []
        self.v2 = []
        self.pim_loss = []

    def store(self, log_prob, entropy, v1, v2, pim_loss):
        self.log_probs.append(log_prob)
        self.entropy.append(entropy)
        self.v1.append(v1)
        self.v2.append(v2)
        self.pim_loss.append(pim_loss)

    def data(self):
        batch_log_probs = torch.cat(self.log_probs, 0)
        batch_entropy = torch.cat(self.entropy, 0)
        batch_v1 = torch.cat(self.v1, 0)
        batch_v2 = torch.cat(self.v2, 0)
        batch_pim_loss = torch.cat(self.pim_loss, 0)
        return (batch_log_probs, batch_entropy, batch_v1, batch_v2, batch_pim_loss)


class Agent(nn.Module):
    def __init__(self,
                 vision_module,
                 path_integration_module: PathIntegrationModule,
                 actor_critic_module: ActorCriticModule
                 ):
        super().__init__()
        self.vision_module = vision_module
        self.path_integration_module = path_integration_module
        self.actor_critic_module = actor_critic_module

        self.reset()
        self.reset_for_training()

    def reset(self):
        self.previous_action_embedding = self.encode_action(None)
        self.current_grid_code = self.path_integration_module.initial_grid_code()
        self.visual_feature_prediction = None
        self.pim_hx, self.pim_cx = self.path_integration_module.initial_hidden_state()
        self.ac_hx, self.ac_cx = self.actor_critic_module.initial_hidden_state()

    def reset_for_training(self):
        self.train_buffer = TrainBuffer()

        self.previous_action_embedding = self.previous_action_embedding.detach()

        self.current_grid_code = self.current_grid_code.detach()
        self.visual_feature_prediction = None
        self.pim_hx = self.pim_hx.detach()
        self.pim_cx = self.pim_cx.detach()

        self.ac_hx = self.ac_hx.detach()
        self.ac_cx = self.ac_cx.detach()

    def get_context(self) -> Any:
        return (self.previous_action_embedding,
                self.current_grid_code,
                self.visual_feature_prediction,
                (self.pim_hx, self.pim_cx),
                (self.ac_hx, self.ac_cx))

    def set_context(self, context) -> Any:
        (self.previous_action_embedding,
         self.current_grid_code,
         self.visual_feature_prediction,
         (self.pim_hx, self.pim_cx),
         (self.ac_hx, self.ac_cx)) = context

    def encode_action(self, action=None):
        act_dim = self.actor_critic_module.action_space_dim
        if action is None:
            return torch.zeros((1, act_dim), dtype=torch.float32)

        assert len(action.shape) == 1
        return F.one_hot(action, num_classes=act_dim).float()

    def get_device(self):
        return next(iter(self.actor_critic_module.parameters())).device

    def send_context_to(self, device):
        self.previous_action_embedding = self.previous_action_embedding.to(
            device)
        self.current_grid_code = self.current_grid_code.to(device)
        if self.visual_feature_prediction is not None:
            self.visual_feature_prediction = self.visual_feature_prediction.to(device)
        self.pim_hx = self.pim_hx.to(device)
        self.pim_cx = self.pim_cx.to(device)
        self.ac_hx = self.ac_hx.to(device)
        self.ac_cx = self.ac_cx.to(device)

    def step(self, obs_tensor, previous_reward_tensor, goal_grid_code_tensor):
        if goal_grid_code_tensor is None:
            goal_grid_code_tensor = torch.zeros_like(self.current_grid_code)

        # all tensors must be provided in batches
        device = self.get_device()
        self.send_context_to(device)

        obs_tensor = obs_tensor.to(device)
        previous_reward_tensor = previous_reward_tensor.to(device)
        goal_grid_code_tensor = goal_grid_code_tensor.to(device)

        with torch.no_grad():
            visual_feature_map = self.vision_module(obs_tensor)[0]

        if self.visual_feature_prediction is not None:
            pim_loss = self.path_integration_module.loss(visual_feature_map,
                                                         self.visual_feature_prediction)
        else:
            pim_loss = torch.zeros(1, device=device)

        v1, v2, dist, (self.ac_hx, self.ac_cx) = self.actor_critic_module(
            visual_feature_map=visual_feature_map,
            previous_action_embedding=self.previous_action_embedding,
            previous_reward=previous_reward_tensor,
            hidden_state=(self.ac_hx, self.ac_cx),
            current_grid_code=self.current_grid_code,
            goal_grid_code=goal_grid_code_tensor,
        )

        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        action_embedding = self.encode_action(action)

        # Update agent's location and head direction
        self.current_grid_code, self.visual_feature_prediction, (
            self.pim_hx, self.pim_cx) = self.path_integration_module(
            visual_feature_map,
            action_embedding,
            (self.pim_hx, self.pim_cx))

        # Store action for next time step
        self.previous_action_embedding = action_embedding

        # Store tensors for loss calculation
        self.train_buffer.store(
            log_prob=log_prob,
            entropy=entropy,
            v1=v1,
            v2=v2,
            pim_loss=pim_loss.view(1),
        )

        return action

    def predict_value(self, obs_tensor, previous_reward_tensor,
                      goal_grid_code_tensor, context):
        # all tensors must be provided in batches
        if goal_grid_code_tensor is None:
            goal_grid_code_tensor = torch.zeros_like(self.current_grid_code)

        # Setup device for inputs
        device = self.get_device()

        previous_action_embedding, current_grid_code, _, (pim_hx, pim_cx), (
            ac_hx, ac_cx) = context
        previous_action_embedding = previous_action_embedding.to(device)
        current_grid_code = current_grid_code.to(device)
        ac_hx = ac_hx.to(device)
        ac_cx = ac_cx.to(device)

        obs_tensor = obs_tensor.to(device)
        previous_reward_tensor = previous_reward_tensor.to(device)
        goal_grid_code_tensor = goal_grid_code_tensor.to(device)

        # Forward pass without gradients
        with torch.no_grad():
            visual_feature_map = self.vision_module(obs_tensor)[0]

            v1, v2, dist, (ac_hx, ac_cx) = self.actor_critic_module(
                visual_feature_map=visual_feature_map,
                previous_action_embedding=previous_action_embedding,
                previous_reward=previous_reward_tensor,
                hidden_state=(ac_hx, ac_cx),
                current_grid_code=current_grid_code,
                goal_grid_code=goal_grid_code_tensor,
            )

            v = torch.min(v1, v2)

        return v

    def compute_loss(self,
                     rewards,
                     dones,
                     next_value,
                     discount_factor,
                     use_gae=True,
                     tau=0.95,
                     value_loss_coef=1.0,
                     policy_loss_coef=1.0,
                     entropy_reg_coef=1.0,
                     grid_layer_wreg_loss_coef=1.0,
                     pim_loss_coef=1.0
                     ):
        device = self.get_device()

        returns = calculate_returns(rewards=rewards,
                                    next_value=next_value,
                                    discount_factor=discount_factor)
        batch_return = torch.tensor(returns, dtype=torch.float32).unsqueeze(
            0).to(device)

        # All tensors have shape of (T, 1)
        # MSE loss against Bellman backup
        (batch_log_probs,
         batch_entropy,
         batch_v1,
         batch_v2,
         batch_pim_loss) = self.train_buffer.data()

        loss_v1 = (batch_return - batch_v1).pow(2).mean()
        loss_v2 = (batch_return - batch_v2).pow(2).mean()
        loss_v = value_loss_coef * (loss_v1 + loss_v2)

        # Policy loss
        batch_value = torch.min(batch_v1.detach(), batch_v2.detach())
        if use_gae:
            advantages = generalized_advantage_estimate(
                rewards=rewards,
                values=batch_value.squeeze(0).cpu().numpy(),
                next_value=next_value,
                discount_factor=discount_factor,
                tau=tau
            )
            batch_advantage = torch.tensor(advantages,
                                           dtype=torch.float32).unsqueeze(0).to(
                device)
        else:
            batch_advantage = batch_return - batch_value
        loss_pi = -policy_loss_coef * (batch_advantage * batch_log_probs).mean()

        # Entropy-regularization
        loss_entropy = -entropy_reg_coef * batch_entropy.mean()

        # Path integration losses
        # Weight regularization loss
        loss_grid_l2 = grid_layer_wreg_loss_coef * self.path_integration_module.l2_loss()
        # Prediction loss
        loss_pim = pim_loss_coef * batch_pim_loss.sum()

        # Total loss
        loss = loss_v + loss_pi + loss_entropy + loss_grid_l2 + loss_pim

        # Useful info for logging
        info = dict(
            LossV=loss_v.detach().cpu().numpy(),
            LossPi=loss_pi.detach().cpu().numpy(),
            LossEntropy=loss_entropy.detach().cpu().numpy(),
            LossGridL2=loss_grid_l2.detach().cpu().numpy(),
            LossPIM=loss_pim.detach().cpu().numpy(),
            Value=batch_value.detach().cpu().numpy(),
            LogPi=batch_log_probs.detach().cpu().numpy(),
        )

        return loss, info

    def act(self, obs, previous_reward, goal_grid_code, deterministic=False):
        # All tensors must be provided in batches
        device = self.get_device()

        self.send_context_to(device)

        obs = torch.tensor(obs).float().unsqueeze(0).to(device)
        previous_reward = torch.tensor([previous_reward]).float().unsqueeze(
            0).to(device)
        if goal_grid_code is None:
            goal_grid_code = torch.zeros_like(self.current_grid_code).to(device)
        else:
            goal_grid_code = torch.tensor(goal_grid_code).float().to(device)

        with torch.no_grad():
            visual_feature_map = self.vision_module(obs)[0]

            v1, v2, dist, (self.ac_hx, self.ac_cx) = self.actor_critic_module(
                visual_feature_map=visual_feature_map,
                previous_action_embedding=self.previous_action_embedding,
                previous_reward=previous_reward,
                hidden_state=(self.ac_hx, self.ac_cx),
                current_grid_code=self.current_grid_code,
                goal_grid_code=goal_grid_code,
            )

            if deterministic:
                action = dist.probs.argmax(-1)
            else:
                action = dist.sample()

            action_embedding = self.encode_action(action)

            # Update agent's location and head direction
            self.current_grid_code, _, (self.pim_hx, self.pim_cx) = \
                self.path_integration_module(visual_feature_map,
                                             action_embedding,
                                             (self.pim_hx, self.pim_cx))
            # Store action for next time step
            self.previous_action_embedding = action_embedding

        return action.cpu().item()


def make_agent(env,
               vision_model_checkpoint_filepath,
               pim_checkpoint_filepath=None,
               pim_lstm_hidden_size=256,
               grid_layer_size=256,
               grid_layer_dropout_rate=0.5,
               ac_lstm_hidden_size=256,
               actor_hidden_sizes=(256,),
               critic_hidden_sizes=(256,)
               ):
    image_shape = env.observation_space.shape
    act_dim = env.action_space.n

    vision_module = make_vision_module('VQVAE', vision_model_checkpoint_filepath)

    trainable(vision_module, False)
    if pim_checkpoint_filepath:
        path_integration_module = torch.load(pim_checkpoint_filepath, map_location=DEVICE)
    else:
        path_integration_module = PathIntegrationModule(
            visual_feature_size=vision_module.output_shape[0],
            action_space_dim=act_dim,
            lstm_hidden_size=pim_lstm_hidden_size,
            grid_layer_size=grid_layer_size,
            grid_layer_dropout_rate=grid_layer_dropout_rate
        )

    actor_critic_module = ActorCriticModule(
        visual_feature_size=vision_module.output_shape[0],
        grid_code_size=path_integration_module.output_shapes['grid_activation'][0],
        action_space_dim=act_dim,
        lstm_hidden_size=ac_lstm_hidden_size,
        actor_hidden_sizes=list(actor_hidden_sizes),
        critic_hidden_sizes=list(critic_hidden_sizes)
    )

    agent = Agent(
        vision_module=vision_module,
        path_integration_module=path_integration_module,
        actor_critic_module=actor_critic_module
    )

    # Count variables (pro-tip: try to get a feel for how different size networks behave!)
    print('\nNumber of parameters')
    print('-' * 32)
    print(f'Vision module: \t {nn_utils.count_vars(vision_module):d}')
    print(
        f'Path integration module: \t {nn_utils.count_vars(path_integration_module):d}')
    print(
        f'Actor critic module: \t {nn_utils.count_vars(actor_critic_module):d}')

    return agent
