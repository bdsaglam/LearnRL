import torch
import torch.nn as nn
from torch.nn import functional as F

from spinup.utils.nn_ext import MLP


class GridPredictor(nn.Module):
    def __init__(self, dim_grid, dim_feature_map, dims_hidden=tuple()):
        super().__init__()
        self.fcs = MLP(sizes=[dim_grid, *dims_hidden, dim_feature_map],
                       activate_final=False)

    def forward(self, feature_map, grid_activation):
        return feature_map + self.fcs(grid_activation)


class PathIntegrationModule(nn.Module):
    def __init__(self,
                 visual_feature_size,
                 action_space_dim,
                 lstm_hidden_size=256,
                 grid_layer_size=256,
                 grid_layer_dropout_rate=0.5,
                 predictor_hidden_sizes=tuple()):
        super().__init__()
        self.visual_feature_size = visual_feature_size
        self.action_space_dim = action_space_dim
        self.grid_layer_size = grid_layer_size

        self.lstm_cell = nn.LSTMCell(
            visual_feature_size + action_space_dim,
            hidden_size=lstm_hidden_size)
        self.grid_layer = nn.Linear(lstm_hidden_size, grid_layer_size,
                                    bias=False)
        self.dropout = nn.Dropout(grid_layer_dropout_rate)
        self.predictor = GridPredictor(dim_grid=grid_layer_size,
                                       dim_feature_map=visual_feature_size,
                                       dims_hidden=predictor_hidden_sizes)
        self.output_shapes = dict(
            grid_activation=grid_layer_size,
            visual_feature_prediction=visual_feature_size,
        )

    def forward(self, visual_feature_map, action_embedding, hidden_state):
        assert len(visual_feature_map.shape) == 2
        assert len(action_embedding.shape) == 2

        hx, cx = hidden_state
        x = torch.cat((visual_feature_map, action_embedding), 1)
        hx, cx = self.lstm_cell(x, (hx, cx))
        grid_activations = self.dropout(self.grid_layer(hx))
        predicted_visual_feature_map = self.predictor(visual_feature_map, grid_activations)
        return grid_activations, predicted_visual_feature_map, (hx, cx)

    def get_device(self):
        return self.grid_layer.weight.device

    def initial_hidden_state(self):
        device = self.get_device()

        hx = torch.zeros(1, self.lstm_cell.hidden_size, device=device)
        cx = torch.zeros_like(hx)
        return hx, cx

    def initial_grid_code(self):
        device = self.get_device()
        return torch.zeros(1, self.grid_layer_size, device=device)

    def l2_loss(self):
        return self.grid_layer.weight.norm(2)

    def loss(self, visual_feature_map, predicted_vfm):
        return F.mse_loss(predicted_vfm, visual_feature_map)
