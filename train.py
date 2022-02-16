import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import AlexNet
import torchvision.datasets as datasets
from utils import load_checkpoint, save_checkpoint
import torch.nn.functional as F


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

# DataLoader
train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)

# Initializing model

model = AlexNet()

# Optim
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)  # Paper used SGD

# Multiply lr by 1 / 10 every 30 epochs
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

if LOAD_MODEL:
    load_checkpoint("CHECKPOINT.pt", model, optimizer, LEARNING_RATE)


def train(model, optimizer, dataloader, epoch):
    print("---TRAINING---")
    total_steps = 1
    model.train()
    if epoch % 5 == 0:
        # Save checkpoint
        save_checkpoint(model, optimizer, filename="CHECKPOINT.pt")
    for images, classes in dataloader:
        images, classes = images.to(device), classes.to(device)

        # Calcualte loss
        output = model(images)
        loss = F.cross_entropy(output, classes)

        # Update  parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if total_steps % 100 == 0:
            with torch.no_grad():
                # also print and save parameter values
                print("---")
                for name, parameter in model.named_parameters():
                    # Print grad of the parameters
                    if parameter.grad is not None:
                        avg_grad = torch.mean(parameter.grad)
                        print(f"Grad Avg. {avg_grad}")
                    # Print parameters values
                    if parameter.data is not None:
                        avg_weight = torch.mean(parameter.data)
                        print(f"Parameter Avg. {avg_weight}")
                    # Print epoch and loss
                    print(f"Epoch [{epoch}/{EPOCHS}] Loss: {loss:.4f}")

        total_steps += 1


for epoch in range(EPOCHS):
    lr_scheduler.step()
    train(model, optimizer, train_dataloader, epoch)
