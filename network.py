from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import torch.nn as nn

class Conv(nn.Module):
    def __init__(self, num_classes=10):
        super(Conv, self).__init__()  # Mandatory call to super
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(5, 5), stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=1),
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5), stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=1),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(10368, 100),
            nn.ReLU(),
            nn.Linear(100, num_classes),
        )
    
    def forward(self, x):
        conv_out = self.conv_layers(x)
        print(conv_out.shape)
        fc_out = self.fc_layers(conv_out.view(-1, 10368))
        return fc_out