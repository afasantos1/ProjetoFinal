import os
import time
import torch
from PIL import Image
import pandas as pd
from skimage import io
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset

# --- Define Data Transformations ---
data_transforms = transforms.Compose([
    transforms.ToPILImage(),  # Converts numpy array to PIL Image
    transforms.ToTensor(),    # Converts to tensor and scales [0,1]
    transforms.Normalize(mean=[0.5, 0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5, 0.5])  # Normalize
])

# --- Define the Neural Network ---
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 497)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.adaptive_avg_pool2d(x, (5, 5))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# --- Define the Custom Dataset ---
class CoinsDataset(Dataset):
    """Coins dataset."""
    def __init__(self, csv_file, root_dir, transform=None):
        self.coins_data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.coins_data)

    def __getitem__(self, idx):
        label = int(self.coins_data.iloc[idx, 0])  # <-- FIX HERE
        img_name = os.path.join(self.root_dir, self.coins_data.iloc[idx, 1])
        image = io.imread(img_name)

        if self.transform:
            image = self.transform(image)

        return image, label  # Returns (image, label)

# ------------------------ Main ---------------------------
# Set parameters and file paths
csv_file = 'image_data.csv'
root_dir = '../Imagens_tratadas2'
alt_csv_file = '../alt_imgs/file_list.csv'
alt_root_dir = '../alt_imgs'

# Load main dataset (for training and original testing)
# Load the original and alternative datasets
orig_dataset = CoinsDataset(csv_file, root_dir, transform=data_transforms)
alt_dataset = CoinsDataset('../alt_imgs2/dados_teste2.csv', '../alt_imgs2', transform=data_transforms)

# --- Third Dataset (New One You Want to Add) ---
third_csv_file = '../imagens_treino/Belgium_2009/dados_belgium_2009.csv'
third_root_dir = '../imagens_treino/Belgium_2009/replicas_resized'

third_dataset = CoinsDataset(third_csv_file, third_root_dir, transform=data_transforms)


# Create label list from the dataset
# Get labels from both datasets
orig_labels = orig_dataset.coins_data.iloc[:, 0].values
alt_labels = alt_dataset.coins_data.iloc[:, 0].values
third_labels = third_dataset.coins_data.iloc[:, 0].values
# Combine all datasets and labels
labels = np.concatenate([orig_labels, alt_labels, third_labels])
dataset = ConcatDataset([orig_dataset, alt_dataset, third_dataset])

# Split into training (90%) and testing (10%) sets
train_indices, test_indices = train_test_split(
    np.arange(len(dataset)), test_size=0.1, stratify=labels, random_state=42
)

# Create subset datasets for training and original test set
train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)

# Define DataLoaders for training and original test set
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)


# --- Set up the Model, Loss, and Optimizer ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0003)

# --- Train the Model ---
num_epochs = 15
t0 = time.time()
print(f"Min label: {labels.min().item()}, Max label: {labels.max().item()}")


for epoch in range(num_epochs):
    net.train()  # Set model to training mode
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

print("Finished Training")

# Save model
PATH = './Prototype.pth'
torch.save(net.state_dict(), PATH)
print(f"Model saved to {PATH}")

# --- Define a Function to Evaluate the Model ---
def evaluate_model(loader, description="Test"):
    net.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"{description} Accuracy: {accuracy:.2f}%")
    print(f"Correct: {correct} vs Total: {total}")

# --- Evaluate on the Original Test Group ---
evaluate_model(test_loader, description="Original Test Group")

# --- Alternative Test with alt_imgs/file_list.csv ---
alt_dataset = CoinsDataset(alt_csv_file, alt_root_dir, transform=data_transforms)
alt_loader = DataLoader(alt_dataset, batch_size=8, shuffle=False, num_workers=4)
evaluate_model(alt_loader, description="Alternative Test Group")
print(f"Total training time: {time.time() - t0:.2f} seconds")



del train_loader, test_loader, alt_loader
torch.cuda.empty_cache()  # Optional, for GPU memory cleanup

import gc
gc.collect()
