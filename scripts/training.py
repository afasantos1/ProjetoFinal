"""
@file training.py
@brief Este script implementa o processo de treino de uma Rede Neural Convolucional (RNC)
       para classificação de imagens, utilizando PyTorch. Inclui a definição da rede,
       dataset personalizado, carregamento e divisão de dados, treino e avaliação. Foi utilizado
       na fase inicial do projeto quando o objetivo era criar uma RNC do zero.
"""

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
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from torchvision import transforms
from sklearn.model_selection import train_test_split

# -------------------------------------------
# Transformações a aplicar nas imagens
# -------------------------------------------
data_transforms = transforms.Compose([
    transforms.ToPILImage(),  # Converte array numpy para imagem PIL
    transforms.ToTensor(),    # Converte imagem PIL para tensor do PyTorch
    transforms.Normalize(mean=[0.5, 0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5, 0.5])  # Normaliza para intervalo [-1, 1]
])

# -------------------------------------------
# Definição da Rede Neuronal Convolucional
# -------------------------------------------
class Net(nn.Module):
    """
    @brief Uma Rede Neural Convolucional (RNC) simples para classificação de imagens.
    """
    def __init__(self):
        """
        @brief Inicializa as camadas da rede.
        Contém duas camadas convolucionais, seguidas de pooling e três camadas lineares.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(4, 6, 5)     # Primeira camada convolucional (entrada com 4 canais)
        self.pool = nn.MaxPool2d(2, 2)      # Camada de pooling com janela 2x2
        self.conv2 = nn.Conv2d(6, 16, 5)    # Segunda camada convolucional
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # Primeira camada totalmente ligada
        self.fc2 = nn.Linear(120, 84)          # Segunda camada totalmente ligada
        self.fc3 = nn.Linear(84, 497)          # Camada de saída (497 classes)

    def forward(self, x):
        """
        @brief Define a passagem direta (forward) da rede.
        @param x Tensor de entrada (batch de imagens).
        @return Tensor de saída (logits de cada classe).
        """
        x = self.pool(F.relu(self.conv1(x)))      # Conv -> ReLU -> Pool
        x = self.pool(F.relu(self.conv2(x)))      # Conv -> ReLU -> Pool
        x = F.adaptive_avg_pool2d(x, (5, 5))      # Redimensiona para 5x5 (garante compatibilidade)
        x = torch.flatten(x, 1)                   # Achata para vetor (mantém dimensão do batch)
        x = F.relu(self.fc1(x))                   # FC1 -> ReLU
        x = F.relu(self.fc2(x))                   # FC2 -> ReLU
        x = self.fc3(x)                           # FC3 (saída final)
        return x

# -------------------------------------------
# Dataset personalizado para imagens de moedas
# -------------------------------------------
class CoinsDataset(Dataset):
    """
    @brief Dataset personalizado para carregar imagens de moedas a partir de ficheiro CSV.
    """
    def __init__(self, csv_file, root_dir, transform=None):
        """
        @param csv_file Caminho para o ficheiro CSV com as etiquetas e nomes dos ficheiros.
        @param root_dir Diretório onde se encontram as imagens.
        @param transform Transformações a aplicar nas imagens.
        """
        self.coins_data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        """
        @brief Devolve o número de amostras no dataset.
        """
        return len(self.coins_data)

    def __getitem__(self, idx):
        """
        @brief Recupera uma imagem e a sua etiqueta pelo índice.
        @param idx Índice da imagem.
        @return Uma tupla com (imagem transformada, etiqueta)
        """
        label = int(self.coins_data.iloc[idx, 0])
        img_name = os.path.join(self.root_dir, self.coins_data.iloc[idx, 1])
        image = io.imread(img_name)

        if self.transform:
            image = self.transform(image)

        return image, label

# -------------------------------------------
# Carregamento dos datasets
# -------------------------------------------
csv_file = 'image_data.csv'
root_dir = '../Imagens_tratadas2'
alt_csv_file = '../alt_imgs/file_list.csv'
alt_root_dir = '../alt_imgs'

# Carrega os três conjuntos de dados: original, alternativo e o novo
orig_dataset = CoinsDataset(csv_file, root_dir, transform=data_transforms)
alt_dataset = CoinsDataset('../alt_imgs2/dados_teste2.csv', '../alt_imgs2', transform=data_transforms)
third_dataset = CoinsDataset('../imagens_treino/Belgium_2009/dados_belgium_2009.csv', '../imagens_treino/Belgium_2009/replicas_resized', transform=data_transforms)

# Junta todas as etiquetas dos datasets
orig_labels = orig_dataset.coins_data.iloc[:, 0].values
alt_labels = alt_dataset.coins_data.iloc[:, 0].values
third_labels = third_dataset.coins_data.iloc[:, 0].values
labels = np.concatenate([orig_labels, alt_labels, third_labels])

# Concatena os datasets num único conjunto combinado
dataset = ConcatDataset([orig_dataset, alt_dataset, third_dataset])

# Divide os dados em treino (90%) e teste (10%) com estratificação
train_indices, test_indices = train_test_split(
    np.arange(len(dataset)), test_size=0.1, stratify=labels, random_state=42
)

# Cria subconjuntos para treino e teste
train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)

# Define os DataLoaders para treino e teste
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)

# -------------------------------------------
# Inicialização do modelo, função de perda e otimizador
# -------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0003)

# -------------------------------------------
# Treino da rede
# -------------------------------------------
num_epochs = 15
t0 = time.time()
print(f"Etiqueta mínima: {labels.min().item()}, Etiqueta máxima: {labels.max().item()}")

for epoch in range(num_epochs):
    net.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Época [{epoch+1}/{num_epochs}], Perda média: {running_loss/len(train_loader):.4f}")

print("Treinamento concluído")

# Guarda o modelo treinado
PATH = './Prototype.pth'
torch.save(net.state_dict(), PATH)
print(f"Modelo guardado em {PATH}")

# -------------------------------------------
# Função para avaliar o modelo
# -------------------------------------------
def evaluate_model(loader, description="Teste"):
    """
    @brief Avalia o modelo num conjunto de dados.
    @param loader DataLoader com os dados de teste.
    @param description Nome descritivo do conjunto para impressão.
    """
    net.eval()
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
    print(f"Acurácia ({description}): {accuracy:.2f}%")
    print(f"Corretas: {correct} / Total: {total}")

# Avaliação no conjunto de teste original
evaluate_model(test_loader, description="Conjunto de Teste Original")

# Avaliação com imagens alternativas
alt_dataset = CoinsDataset(alt_csv_file, alt_root_dir, transform=data_transforms)
alt_loader = DataLoader(alt_dataset, batch_size=8, shuffle=False, num_workers=4)
evaluate_model(alt_loader, description="Grupo de Teste Alternativo")

print(f"Tempo total de treino: {time.time() - t0:.2f} segundos")

# -------------------------------------------
# Limpeza de memória (opcional)
# -------------------------------------------
del train_loader, test_loader, alt_loader
torch.cuda.empty_cache()

import gc
gc.collect()
