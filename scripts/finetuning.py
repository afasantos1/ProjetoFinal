import os
import time
import torch
import numpy as np
import pandas as pd
from PIL import Image
from skimage import io
from collections import Counter
from sklearn.model_selection import train_test_split

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from torchvision import transforms, models

# -------------------------------
# Configurações Gerais e Paths
# -------------------------------
BATCH_SIZE = 8
NUM_EPOCHS = 15
MODEL_PATH = './mobilenet_v2-finetuned.pth'
CSV_ROOT_PAIRS = [
    ('image_data.csv', '../Imagens_tratadas2'),
    ('../alt_imgs/file_list.csv', '../alt_imgs'),
    ('../imagens_treino/Belgium_2009/dados_belgium_2009.csv', '../imagens_treino/Belgium_2009/replicas_resized'),
    ('../imagens_treino/GR2004/Greece.csv', '../imagens_treino/GR2004/'),
    ('../imagens_treino/FR2007 treaty/France.csv', '../imagens_treino/FR2007 treaty/'),
    ('../imagens_treino/NL2013-2/Netherlands.csv', '../imagens_treino/NL2013-2/'),
    ('../imagens_treino/PT2016-1/Portugal.csv', '../imagens_treino/PT2016-1/'),
    ('../imagens_treino/PT2016-2/Portugal2.csv', '../imagens_treino/PT2016-2/'),
]

starred_indices = [3, 4, 5, 6, 7]

# -------------------------------
# Transformações de Dados
# -------------------------------
data_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -------------------------------
# Dataset Personalizado
# -------------------------------
class CoinsDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.coins_data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.coins_data)

    def __getitem__(self, idx):
        label = int(self.coins_data.iloc[idx, 0])
        img_path = os.path.join(self.root_dir, self.coins_data.iloc[idx, 1])
        img = io.imread(img_path)

        if img.shape[2] == 4:
            img = img[:, :, :3]

        if self.transform:
            img = self.transform(img)

        return img, label

# -------------------------------
# Carregamento e Divisão de Dados
# -------------------------------
global_train_datasets = []
individual_test_sets = {}
label_counts_all = Counter()

for idx, (csv, root) in enumerate(CSV_ROOT_PAIRS):
    ds = CoinsDataset(csv, root, transform=data_transforms)
    labels = ds.coins_data.iloc[:, 0].values
    label_counts = Counter(labels)
    label_counts_all.update(label_counts)

    valid_indices = [i for i, label in enumerate(labels) if label_counts[label] >= 2]
    filtered_labels = [labels[i] for i in valid_indices]

    if idx in starred_indices:
        train_idx, test_idx = train_test_split(
            list(range(len(valid_indices))),
            test_size=0.25,
            stratify=filtered_labels,
            random_state=42
        )
        train_idx = [valid_indices[i] for i in train_idx]
        test_idx = [valid_indices[i] for i in test_idx]

        global_train_datasets.append(Subset(ds, train_idx))
        individual_test_sets[csv] = Subset(ds, test_idx)
    else:
        global_train_datasets.append(ds)

train_dataset = ConcatDataset(global_train_datasets)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# -------------------------------
# Configuração do Modelo
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
model = models.mobilenet_v2(weights=weights)

for param in model.features.parameters():
    param.requires_grad = False

num_classes = sum(1 for v in label_counts_all.values() if v >= 2)
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, num_classes)
model = model.to(device)

# -------------------------------
# Treino
# -------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4)

print(f"Classes válidas: {num_classes} | Treinando com {len(train_dataset)} imagens")

start_time = time.time()
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Loss: {avg_loss:.4f}")

print(f"Treinamento concluído em {(time.time() - start_time):.2f} segundos.")
torch.save(model.state_dict(), MODEL_PATH)
print(f"Modelo salvo em {MODEL_PATH}")

# -------------------------------
# Avaliação
# -------------------------------
def evaluate(loader, description="Avaliação"):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"{description} - Precisão: {accuracy:.2f}% ({correct}/{total})")

# Avaliação por conjunto
for csv_path, test_set in individual_test_sets.items():
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    label = os.path.splitext(os.path.basename(csv_path))[0]
    evaluate(test_loader, f"Conjunto de Teste - {label}")
