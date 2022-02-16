import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import AlexNet
import torchvision.datasets as datasets
from utils import load_checkpoint, save_checkpoint

# HyperParams
LOAD_MODEL = False
device = torch.device("cuda" if torch.cuda.is_available else "cpu")
LEARNING_RATE = 0.01
BATCH_SIZE = 128
IMAGE_SIZE = 227
CHANNELS_IMG = 3
EPOCHS = 90
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
