import argparse

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torchvision import datasets, transforms

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', '-bs', type=int, default=128)
args = parser.parse_args()
