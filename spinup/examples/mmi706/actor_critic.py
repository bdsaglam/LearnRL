from copy import deepcopy
from typing import List

import torch
import torch.nn as nn

from spinup.core.approximators import MLPCategoricalActor, MLPVFunction


class ActorCriticModule(nn.Module):
    def __init__(self,
                 visual_feature_size,
                 grid_code_size,
                 action_space_dim,
                 lstm_hidden_size,
                 actor_hidden_sizes: List[int],
                 critic_hidden_sizes: List[int]):
        super().__init__()

        self.action_space_dim = action_space_dim

        input_size = 2 * grid_code_size + visual_feature_size + action_space_dim + 1  # 1 for reward
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size=lstm_hidden_size)
        self.actor = MLPCategoricalActor(lstm_hidden_size,
                                         action_space_dim,
                                         hidden_sizes=actor_hidden_sizes)
        self.critic1 = MLPVFunction(lstm_hidden_size,
                                    hidden_sizes=critic_hidden_sizes)
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
        x = torch.cat(
            (goal_grid_code, current_grid_code, visual_feature_map, previous_action_embedding, previous_reward),
            dim=1
        )
        hx, cx = self.lstm_cell(x, (hx, cx))
        v1 = self.critic1(hx)
        v2 = self.critic2(hx)
        dist = self.actor(hx)

        return v1, v2, dist, (hx, cx)

    def initial_hidden_state(self):
        hx = torch.zeros(1, self.lstm_cell.hidden_size, dtype=torch.float32)
        cx = torch.zeros_like(hx)
        return hx, cx
