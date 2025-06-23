import os
import torch
from skimage import io
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F

# --- Define Data Transformations ---
data_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5, 0.5])
])

# --- Define the Neural Network (Must match the trained one) ---
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
    def __init__(self, csv_file, root_dir, transform=None):
        self.coins_data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.coins_data)

    def __getitem__(self, idx):
        label = self.coins_data.iloc[idx, 0]
        img_name = os.path.join(self.root_dir, self.coins_data.iloc[idx, 1])
        image = io.imread(img_name)
        if self.transform:
            image = self.transform(image)
        return image, label

# --- Load the Model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)
model.load_state_dict(torch.load("./Prototype.pth", map_location=device))
model.eval()
print("Model loaded successfully.")

# --- Evaluation Function ---
def evaluate_model(loader, description="Test"):
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"{description} Accuracy: {accuracy:.2f}%")
    print(f"Correct: {correct} / {total}")

# --- Load Datasets and Evaluate ---
# Original test group
original_csv = 'image_data.csv'
original_root = '../Imagens_tratadas2'
original_dataset = CoinsDataset(original_csv, original_root, transform=data_transforms)
original_loader = DataLoader(original_dataset, batch_size=8, shuffle=False, num_workers=4)
evaluate_model(original_loader, description="Original Test Group")

# Alternative test group
alt_csv = '../alt_imgs2/dados_teste2.csv'
alt_root = '../alt_imgs2'
alt_dataset = CoinsDataset(alt_csv, alt_root, transform=data_transforms)
alt_loader = DataLoader(alt_dataset, batch_size=8, shuffle=False, num_workers=4)
evaluate_model(alt_loader, description="Alternative Test Group")