# Libs externes
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torch.optim import Adam
from tqdm.notebook import trange, tqdm
import numpy as np
import functools
import matplotlib.pyplot as plt