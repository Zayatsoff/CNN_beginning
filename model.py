import torch
import torch.nn as nn
import torch.nn.functional as F


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.anet = nn.Sequential(
            # Input: (b x 3 x 227 x 227)
            # Conv 1
            nn.Conv2d(
                in_channels=3, out_channels=96, kernel_size=11, stride=4
            ),  # (b x 96 x 55 x 55)
            nn.LocalResponseNorm(size=5, k=2.0),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 96 x 27 x 27)
            nn.ReLU(),
            # Conv 2
            nn.Conv2d(
                in_channels=96, out_channels=256, kernel_size=5, padding=1
            ),  # (b x 256 x 27 x 27)
            nn.LocalResponseNorm(size=5, k=2.0),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 256 x 13 x 13)
            nn.ReLU(),
            # Conv 3
            nn.Conv2d(
                in_channels=256, out_channels=385, kernel_size=3, padding=1
            ),  # (b x 384 x 13 x 13)
            nn.ReLU(),
            # Conv 4
            nn.Conv2d(
                in_channels=385, out_channels=385, kernel_size=3, padding=1
            ),  # (b x 384 x 13 x 13)
            nn.ReLU(),
            # Conv 5
            nn.Conv2d(
                in_channels=385, out_channels=256, kernel_size=3, padding=1
            ),  # (b x 256 x 13 x 13)
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 256 x 6 x 6)
            nn.ReLU(),
            # Linear 1
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=256 * 6 * 6, out_features=4096),
            nn.ReLU(),
            # Linear 2
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            # Linear 3
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            # Softmax
            nn.Softmax(10),
        )

    def forward(self, x):
        return self.anet(x)
