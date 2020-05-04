from copy import deepcopy
from typing import List

import torch
import torch.nn as nn

from spinup.core.approximators import MLPCategoricalActor, MLPVFunction


class VisionModule(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        c, h, w = input_shape
        self.network = nn.Sequential(
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
            self.output_shape = self.network(torch.rand(1, c, h, w)).squeeze(0).shape

    def forward(self, image):
        return self.network(image)


class PathIntegrationModule(nn.Module):
    def __init__(self,
                 visual_feature_size,
                 action_embedding_size,
                 lstm_hidden_size,
                 grid_layer_size,
                 grid_layer_dropout_rate=0.5):
        super().__init__()
        self.visual_feature_size = visual_feature_size
        self.action_embedding_size = action_embedding_size
        self.grid_layer_size = grid_layer_size

        self.lstm_cell = nn.LSTMCell(visual_feature_size + action_embedding_size, hidden_size=lstm_hidden_size)
        self.grid_layer = nn.Linear(lstm_hidden_size, grid_layer_size, bias=False)
        self.dropout = nn.Dropout(grid_layer_dropout_rate)

        self.output_shape = (grid_layer_size,)

    def forward(self, visual_feature_map, action_embedding, hidden_state):
        assert len(visual_feature_map.shape) == 2
        assert len(action_embedding.shape) == 2
        assert visual_feature_map.shape[0] == action_embedding.shape[0] == 1

        hx, cx = hidden_state
        x = torch.cat((visual_feature_map, action_embedding), 1)
        hx, cx = self.lstm_cell(x, (hx, cx))
        grid_activations = self.dropout(self.grid_layer(hx))
        return grid_activations, (hx, cx)

    def initial_hidden_state(self):
        hx = torch.zeros(1, self.lstm_cell.hidden_size, dtype=torch.float32)
        cx = torch.zeros_like(hx)
        return hx, cx

    def initial_grid_code(self):
        return torch.zeros(1, self.grid_layer_size, dtype=torch.float32)

    def l2_loss(self):
        return self.grid_layer.weight.norm(2)


class ActorCriticModule(nn.Module):
    def __init__(self,
                 visual_feature_size,
                 grid_code_size,
                 action_embedding_size,
                 lstm_hidden_size,
                 actor_hidden_sizes: List[int],
                 critic_hidden_sizes: List[int]):
        super().__init__()

        self.action_embedding_size = action_embedding_size

        input_size = 2 * grid_code_size + visual_feature_size + action_embedding_size + 1  # 1 for reward
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size=lstm_hidden_size)
        self.actor = MLPCategoricalActor(lstm_hidden_size, action_embedding_size, hidden_sizes=actor_hidden_sizes)
        self.critic1 = MLPVFunction(lstm_hidden_size, hidden_sizes=critic_hidden_sizes)
        self.critic2 = deepcopy(self.critic1)

    def forward(self,
                visual_feature_map,
                previous_action_embedding,
                previous_reward,
                hidden_state,
                current_grid_code,
                goal_grid_code,
                ):
        assert len(current_grid_code.shape) == 2
        assert len(goal_grid_code.shape) == 2
        assert len(visual_feature_map.shape) == 2
        assert len(previous_action_embedding.shape) == 2
        assert len(hidden_state) == 2

        hx, cx = hidden_state
        x = torch.cat((goal_grid_code, current_grid_code, visual_feature_map, previous_action_embedding, previous_reward), 1)
        hx, cx = self.lstm_cell(x, (hx, cx))
        v1 = self.critic1(hx)
        v2 = self.critic2(hx)
        dist = self.actor(hx)

        return v1, v2, dist, (hx, cx)

    def initial_hidden_state(self):
        hx = torch.zeros(1, self.lstm_cell.hidden_size, dtype=torch.float32)
        cx = torch.zeros_like(hx)
        return hx, cx
