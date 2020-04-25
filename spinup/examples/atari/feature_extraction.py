from collections import deque

import torch
import torch.nn as nn
import torchvision
from PIL import Image

FRAME_SHAPE = (3, 80, 80)
NUM_RECENT_FRAMES = 4


def preprocess(obs):
    img = Image.fromarray(obs[32:192], mode="RGB")
    c, h, w = FRAME_SHAPE
    transformation = torchvision.transforms.Compose([
        torchvision.transforms.Resize((h, w)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((127, 127, 127), (255, 255, 255))
    ])
    img_tensor = transformation(img)
    return img_tensor


def make_frame_buffer():
    c, h, w = FRAME_SHAPE
    buffer = []
    for _ in range(NUM_RECENT_FRAMES):
        buffer.append(torch.zeros(c, h, w, dtype=torch.float32))

    return deque(buffer, maxlen=NUM_RECENT_FRAMES)


def frames_feature_extractor():
    c, h, w = FRAME_SHAPE

    network = nn.Sequential(
        nn.Conv2d(c * NUM_RECENT_FRAMES, 32, kernel_size=3, stride=1, padding=1),
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
        feature_dim = network(torch.rand(1, c * NUM_RECENT_FRAMES, h, w)).shape

    return network, feature_dim[-1]
