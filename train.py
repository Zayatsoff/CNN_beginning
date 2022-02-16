import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import AlexNet
import torchvision.datasets as datasets
from utils import load_checkpoint, save_checkpoint

# HyperParams
LOAD_MODEL = False
device = torch.device("cuda" if torch.cuda.is_available else "cpu")
LEARNING_RATE = 0.001  # Paper sued 0.01
BATCH_SIZE = 128
IMAGE_SIZE = 227
CHANNELS_IMG = 3
EPOCHS = 90
# MOMENTUM = 0.9 Used by paper
# WEIGHT_DECAY = 5e-4 Used by paper
DOWNLOAD_DATASET = True

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

training_data = datasets.CIFAR10(
    root="data", train=True, download=DOWNLOAD_DATASET, transform=transformation
)

test_data = datasets.CIFAR10(
    root="data", train=False, download=DOWNLOAD_DATASET, transform=transformation
)

# DataLoader
train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

# Initializing model

model = AlexNet()

# Optim
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)  # Paper used SGD

# Multiply lr by 1 / 10 every 30 epochs
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)


# Training
model.train()

if LOAD_MODEL:
    load_checkpoint("CHECKPOINT.pt", model, optimizer, LEARNING_RATE)

total_steps = 1
for epoch in range(EPOCHS):
    lr_scheduler.step()
    for images, classes in train_dataloader:
        images, classes = images.to(device), classes.to(device)
