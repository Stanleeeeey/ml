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
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(5, 5), stride=1, padding=0),
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


def train_conv(model, num_epochs, lr=0.001, device='cuda'):
    model = model.to(device)
    
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        cum_loss = 0
        for (inputs, labels) in train_data_loader:
  
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # original shape is [batch_size, 28, 28] because it's an image of size 28x28
            # inputs = inputs.view(-1, 28 * 28)

            # Do Forward -> Loss Computation -> Backward -> Optimization
            optimizer.zero_grad()
            predictions = model(inputs)


            loss = loss_function(predictions, labels)
            loss.backward()
            optimizer.step()
            
            cum_loss += loss.item()

        print("Epoch %d, Loss=%.4f" % (epoch+1, cum_loss/len(train_data_loader)))
