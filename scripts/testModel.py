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
        self.fc3 = nn.Linear(84, 496)  

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
        img_id = self.coins_data.iloc[idx, 0]  
        img_name = os.path.join(self.root_dir, self.coins_data.iloc[idx, 1])  
        image = io.imread(img_name)  

        if self.transform:
            image = self.transform(image)  

        return image, img_id  # Returns (image, label)

# ------------------------ Main ---------------------------

# Load dataset
csv_file = 'image_data.csv'  
root_dir = '../Imagens_tratadas2'  
dataset = CoinsDataset(csv_file, root_dir, transform=data_transforms)

# Create label list
labels = dataset.coins_data.iloc[:, 0].values  

# Split into training (80%) and testing (20%) sets
train_indices, test_indices = train_test_split(
    np.arange(len(dataset)), test_size=0.1, stratify=labels, random_state=42
)

# Create subset datasets
train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)

# Define DataLoaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)

# Define model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = Net().to(device)

criterion = nn.CrossEntropyLoss()  
optimizer = optim.Adam(net.parameters(), lr=0.001)  

# Train the model
num_epochs = 10
t0 = time.time()

for epoch in range(num_epochs):
    net.train()  # Set model to training mode
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()  # Zero gradients
        outputs = net(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

print("Finished Training")

# Save model
PATH = './Prototype.pth'
torch.save(net.state_dict(), PATH)
print(f"Model saved to {PATH}")

# --- Evaluate Model ---
net.eval()  # Set model to evaluation mode
correct = 0
total = 0

with torch.no_grad():  # No need to track gradients
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)  
        _, predicted = torch.max(outputs, 1)  # Get the predicted class

        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(correct)
print("vs")
print(total)
accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")
print(f"Total training time: {time.time() - t0:.2f} seconds")
# Get unique labels in the train and test sets
unique_train_labels = set(dataset.coins_data.iloc[train_indices, 0])
unique_test_labels = set(dataset.coins_data.iloc[test_indices, 0])

print(f"Number of unique labels in training set: {len(unique_train_labels)}")
print(f"Number of unique labels in test set: {len(unique_test_labels)}")

# Check if any labels are missing in the test set
missing_labels = unique_train_labels - unique_test_labels
print(f"Labels in training but NOT in test: {missing_labels}" if missing_labels else "All labels are present in both sets.")
