import torch
import torch.nn as nn
import torch.nn.functional as F
from capsnet import CapsNet


class CNNVisionModule(nn.Module):
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
        return self.network(image), image

    def reconstruction_loss(self, image, reconstruction):
        return torch.tensor(0, dtype=torch.float32).to(image.device)


class CapsuleVisionModule(nn.Module):
    def __init__(self,
                 input_shape,
                 cnn_out_channels=256,
                 cnn_kernel_size=9,
                 cnn_stride=1,
                 pc_num_capsules=8,
                 pc_out_channels=32,
                 pc_kernel_size=9,
                 pc_stride=2,
                 obj_num_capsules=10,
                 obj_out_channels=16
                 ):
        super().__init__()
        c, h, w = input_shape
        self.network = CapsNet(
            input_shape,
            cnn_out_channels=cnn_out_channels,
            cnn_kernel_size=cnn_kernel_size,
            cnn_stride=cnn_stride,
            pc_num_capsules=pc_num_capsules,
            pc_out_channels=pc_out_channels,
            pc_kernel_size=pc_kernel_size,
            pc_stride=pc_stride,
            obj_num_capsules=obj_num_capsules,
            obj_out_channels=obj_out_channels
        )

        with torch.no_grad():
            obj_vectors, reconstruction, masks = self.network(torch.rand(1, c, h, w))
            self.output_shape = obj_vectors.squeeze(0).view(-1).shape

    def forward(self, image):
        batch_size = image.shape[0]
        obj_vectors, reconstruction, masks = self.network(image)
        return obj_vectors.view(batch_size, -1), reconstruction

    def reconstruction_loss(self, image, reconstruction):
        batch_size = image.shape[0]
        loss = F.mse_loss(
            reconstruction.view(batch_size, -1),
            image.reshape(batch_size, -1)
        )
        return loss
