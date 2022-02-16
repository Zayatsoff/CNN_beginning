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
import torch.nn.functional as F
