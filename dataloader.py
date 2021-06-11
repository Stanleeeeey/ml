import os
import torch
from torchvision import transforms, utils
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt

path = 'flowers'

class FlowerDataset(Dataset):

    def __init__(self, path, transform = None):
        self.path = path
        self.transform = transform
        self.data = [(file_name, label) for label in os.listdir(path) for file_name in os.listdir(os.path.join(path, label))]
        self.label_to_int = {label:idx for idx, label in enumerate(os.listdir(path))}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        datapoint = Image.open(os.path.join(path, self.data[idx][1], self.data[idx][0]))
        label = self.data[idx][1]

        if self.transform:
            datapoint = self.transform(datapoint)

        return datapoint, self.label_to_int[label]

dataset = FlowerDataset(path, transform=transforms.Compose([
            transforms.Resize((200, 200)),
            transforms.ToTensor(),
        ]))
dataloader = DataLoader(dataset)

if __name__ == "__main__":
    image, label = next(iter(dataloader))
    print(label)
    plt.imshow(np.dstack(image[0].numpy()))
    plt.show()