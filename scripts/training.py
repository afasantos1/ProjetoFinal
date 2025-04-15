import os, time
import torch
from PIL import Image
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


data_transforms = transforms.Compose([
    transforms.ToPILImage(),  # Converts the numpy array to a PIL Image, preserving RGBA mode
    transforms.ToTensor(),    # Converts to a tensor and scales the image to [0,1]
    # Optionally, add normalization if needed:
    transforms.Normalize(mean=[0.5, 0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5, 0.5])
])



class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 496)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Shape: [batch, 6, 248, 248]
        x = self.pool(F.relu(self.conv2(x)))  # Shape: [batch, 16, 122, 122]
        x = F.adaptive_avg_pool2d(x, (5, 5))    # Shape: [batch, 16, 5, 5]
        x = torch.flatten(x, 1)                 # Shape: [batch, 16*5*5] = [batch, 400]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x






class Coins_dataset(Dataset):
    """Coins dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.coins_data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.coins_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_id : int = self.coins_data.iloc[idx, 0]
        img_name = os.path.join(self.root_dir,
                                self.coins_data.iloc[idx, 1])
        image = io.imread(img_name)
        sample = {'id': img_id, 'image': image}

        if self.transform:
            sample = self.transform(sample)

        return sample

# ------------------------Main---------------------------
t0 = time.time()
net = Net()
net.to('cuda')
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")



coin_dataset = Coins_dataset(
    csv_file='image_data.csv',
    root_dir='../Imagens_tratadas2',
    transform=lambda sample: {'id': sample['id'], 
                                'image': data_transforms(sample['image'])}
)


# TODO Definir conjunto de teste, ainda nao criado o .csv

dataloader = DataLoader(coin_dataset, batch_size=8,
                shuffle=True, num_workers=4)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)
num_epochs = 10


for epoch in range(num_epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(dataloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs = data['image'].to('cuda')
        labels = data['id'].to('cuda')

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 200 == 199:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
            running_loss = 0.0

print('Finished Training')

PATH = './Prototype.pth'
torch.save(net.state_dict(), PATH)
print('tempo = ', time.time() - t0)
