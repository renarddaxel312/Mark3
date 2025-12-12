import torch
import torch.nn as nn


class MLPInitializer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden=(512, 512)):
        super().__init__()
        layers = []
        dims = [input_dim] + list(hidden) + [output_dim]
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

