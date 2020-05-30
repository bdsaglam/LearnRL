import torch
import torch.nn as nn
from torch.nn import functional as F


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

        self.lstm_cell = nn.LSTMCell(
            visual_feature_size + action_embedding_size,
            hidden_size=lstm_hidden_size)
        self.grid_layer = nn.Linear(lstm_hidden_size, grid_layer_size,
                                    bias=False)
        self.dropout = nn.Dropout(grid_layer_dropout_rate)
        self.predictor = nn.Linear(grid_layer_size, visual_feature_size)
        self.output_shape = (grid_layer_size,)

    def forward(self, visual_feature_map, action_embedding, hidden_state):
        assert len(visual_feature_map.shape) == 2
        assert len(action_embedding.shape) == 2
        assert visual_feature_map.shape[0] == action_embedding.shape[0] == 1

        hx, cx = hidden_state
        x = torch.cat((visual_feature_map, action_embedding), 1)
        hx, cx = self.lstm_cell(x, (hx, cx))
        grid_activations = self.dropout(self.grid_layer(hx))
        prediction = self.predictor(grid_activations)
        return grid_activations, prediction, (hx, cx)

    def initial_hidden_state(self):
        hx = torch.zeros(1, self.lstm_cell.hidden_size, dtype=torch.float32)
        cx = torch.zeros_like(hx)
        return hx, cx

    def initial_grid_code(self):
        return torch.zeros(1, self.grid_layer_size, dtype=torch.float32)

    def l2_loss(self):
        return self.grid_layer.weight.norm(2)

    def loss(self, visual_feature_map, predicted_vfm):
        return F.mse_loss(predicted_vfm, visual_feature_map)
