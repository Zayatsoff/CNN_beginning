import torch
import torch.optim as optim
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import AlexNet
import torchvision.datasets as datasets
from utils import load_checkpoint, train_classifier


# HyperParams
LOAD_MODEL = True
device = torch.device("cuda" if torch.cuda.is_available else "cpu")
LEARNING_RATE = 0.001  # Paper sued 0.01
BATCH_SIZE = 128
IMAGE_SIZE = 224
CHANNELS_IMG = 3
EPOCHS = 70
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

model = AlexNet().to(device)

# Optim
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)  # Paper used SGD

# Multiply lr by 1 / 10 every 30 epochs
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

if LOAD_MODEL:
    load_checkpoint("AlexNet_CIFAR10\\checkpoint.pt", model, optimizer, LEARNING_RATE)


log_dict = train_classifier(
    num_epochs=EPOCHS,
    model=model,
    optimizer=optimizer,
    device=device,
    train_loader=train_dataloader,
    valid_loader=test_dataloader,
    logging_interval=50,
)
