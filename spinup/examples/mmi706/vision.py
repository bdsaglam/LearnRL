import torch
import torch.nn as nn


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


class VQVAEVisionModule(nn.Module):
    def __init__(self, vqvae):
        super().__init__()
        self.vqvae = vqvae

        with torch.no_grad():
            image = torch.rand(1, *vqvae.input_shape)
            self.output_shape = self.forward(image)[0].squeeze(0).shape

    def forward(self, image):
        batch_size = image.shape[0]
        res = self.vqvae(image)
        return res.quantized.view(batch_size, -1), res.vq_loss, res.rec

    def loss(self, image, result):
        quantized, vq_loss, rec = result
        rec_loss = self.vqvae.reconstruction_loss(rec, image)
        return rec_loss + vq_loss


def make_vision_module(model_type, checkpoint_filepath, device=torch.device('cpu')):
    if model_type == 'VQVAE':
        vqvae = torch.load(checkpoint_filepath, map_location=device)
        return VQVAEVisionModule(vqvae)

    raise ValueError("Unexpected model type")
