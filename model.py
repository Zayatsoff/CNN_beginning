import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv_layers = nn.Sequential(
            # Input: (b x 3 x 64 x 64)
            # Conv 1
            nn.Conv2d(
                in_channels=3, out_channels=64, kernel_size=11, stride=4, padding=2
            ),  # (b x 32 x 48 x 48)
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, k=2.0),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 32 x 24 x 24)
            # Conv 2
            nn.Conv2d(
                in_channels=64, out_channels=192, kernel_size=3, padding=1
            ),  # (b x 64 x 12 x 12)
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, k=2.0),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 64 x 6 x 6)
            # Conv 3
            nn.Conv2d(
                in_channels=192, out_channels=384, kernel_size=3, padding=1
            ),  # (b x 128 x 6 x 6)
            nn.ReLU(),
            # Conv 4
            nn.Conv2d(
                in_channels=384, out_channels=256, kernel_size=3, padding=1
            ),  # (b x 256 x 6 x 6)
            nn.ReLU(),
            # Conv 5
            nn.Conv2d(
                in_channels=256, out_channels=256, kernel_size=3, padding=1
            ),  # (b x 256 x 6 x 6)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 256 x 6 x 6)
            # nn.Flatten(),
        )
        self.lin_layers = nn.Sequential(
            # Linear 1
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256 * 6 * 6, out_features=4096),
            nn.ReLU(),
            # Linear 2
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            # Linear 3
            nn.Linear(in_features=4096, out_features=10),
            # nn.ReLU(),
            # Softmax
            # nn.Softmax(),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # flatten
        return self.lin_layers(x)
