import torch
import torch.nn as nn
import torch.nn.functional as F


class AlexNet(nn.Module):
    def __init__(self, channels_img, features_d):
        super(AlexNet, self).__init__()
        self.anet = nn.Sequential(
            # Input: 224*224*3
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),
            F.local_response_norm(),
        )

    def forward(self, x):
        return self.anet(x)
