import torch
import torch.nn as nn
import torch.nn.functional as F
from capsnet import CapsNet


class MockVisionModule(nn.Module):
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
            image = torch.rand(1, c, h, w)
            self.output_shape = self.network(image).squeeze(0).shape

    def forward(self, image):
        return self.network(image), image

    def loss(self, image, result):
        return torch.zeros_like(image)


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
            image = torch.rand(1, *input_shape)
            obj_vectors, reconstruction, masks = self.network(image)
            self.output_shape = obj_vectors.squeeze(0).view(-1).shape

    def forward(self, image):
        batch_size = image.shape[0]
        obj_vectors, reconstruction, masks = self.network(image)
        return obj_vectors.view(batch_size, -1), reconstruction

    def loss(self, image, result):
        batch_size = image.shape[0]
        _, reconstruction = result
        loss = F.mse_loss(
            reconstruction.view(batch_size, -1),
            image.reshape(batch_size, -1)
        )
        return loss


class VQVAEVisionModule(nn.Module):
    def __init__(self, vqvae):
        super().__init__()
        self.vqvae = vqvae

        with torch.no_grad():
            image = torch.rand(1, vqvae.input_shape)
            self.output_shape = self.forward(image).squeeze(0).shape

    def forward(self, image):
        batch_size = image.shape[0]
        res = self.vqvae.encode(image)
        return res.quantized.view(batch_size, -1), res.vq_loss, res.rec

    def loss(self, image, result):
        quantized, vq_loss, rec = result
        rec_loss = self.vqvae.reconstruction_loss(rec, image)
        return rec_loss + vq_loss


def make_vision_module(model_type, checkpoint_filepath, device):
    if model_type == 'VQVAE':
        vqvae = torch.load(checkpoint_filepath, map_location=device)
        return VQVAEVisionModule(vqvae)

    if model_type == 'Capsule':
        return torch.load(checkpoint_filepath, map_location=device)
