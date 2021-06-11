from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from dataloader import dataloader
import random
import torch.nn as nn

class Conv(nn.Module):
    def __init__(self, num_classes=10):
        super(Conv, self).__init__()  # Mandatory call to super
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(5, 5), stride=1, padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=1),
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5), stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=1),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(1280000, 100),
            nn.ReLU(),
            nn.Linear(100, num_classes),
        )

    def forward(self, x):
        conv_out = self.conv_layers(x)
     
        fc_out = self.fc_layers(conv_out.view(-1, 1280000))
        return fc_out


def train_conv(model, num_epochs, lr=0.001, device='cuda'):
    model = model.to(device)
    
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        cum_loss = 0
        for (inputs, labels) in dataloader:
           
            inputs = inputs.to(device)
            labels = labels.to(device)
            

            optimizer.zero_grad()
            predictions = model(inputs)


            loss = loss_function(predictions, labels)
            loss.backward()
            optimizer.step()
            
            cum_loss += loss.item()
            #print('working' + random.choice(['/','|']))
         

        print("Epoch %d, Loss=%.4f" % (epoch+1, cum_loss/len(dataloader)))


def calculate_accuracy_conv(model, device = 'cuda', data_loader=dataloader):
    model = model.to(device)

    correct = 0
    total = 0
    for (inputs, labels) in data_loader:

        inputs = inputs.to(device)  # 64 x 28 x 28
        labels = labels.to(device)

        # inputs = inputs.view(-1, 28 * 28) # 64 x 784

        predictions = model(inputs)  # 64 x 10
        
        correct += (predictions.argmax(dim=1) == labels).sum()
        total += len(labels)

    return correct / total

model = Conv(num_classes=6)
train_conv(
    model = model,
    num_epochs = 10,
)

print(calculate_accuracy_conv(

    model = model,

))