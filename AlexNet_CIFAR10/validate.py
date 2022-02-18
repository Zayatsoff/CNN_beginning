import torch
import torch.optim as optim
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import AlexNet
import torchvision.datasets as datasets
from utils import load_checkpoint, validate_classifier


# HyperParams
LOAD_MODEL = True
device = torch.device("cuda" if torch.cuda.is_available else "cpu")
LEARNING_RATE = 0.001  # Paper sued 0.01
BATCH_SIZE = 128
IMAGE_SIZE = 224
CHANNELS_IMG = 3
# MOMENTUM = 0.9 Used by paper
# WEIGHT_DECAY = 5e-4 Used by paper
DOWNLOAD_DATASET = False

# Dataset
transformation = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]
        ),
    ]
)
test_data = datasets.CIFAR10(
    root="data", train=False, download=DOWNLOAD_DATASET, transform=transformation
)

# DataLoader
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

# Initializing model

model = AlexNet().to(device)

# Optim
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)  # Paper used SGD

if LOAD_MODEL:
    load_checkpoint("AlexNet_CIFAR10\\checkpoint.pt", model, optimizer, LEARNING_RATE)


log_dict = validate_classifier(
    model=model,
    device=device,
    valid_loader=test_dataloader,
)
