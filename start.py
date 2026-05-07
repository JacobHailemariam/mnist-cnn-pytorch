import torch, torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307))])