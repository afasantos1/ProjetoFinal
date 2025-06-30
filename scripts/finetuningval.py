"""
@file finetuningval.py
@brief Este script carrega um modelo MobileNetV2 previamente treinado e avalia o seu desempenho
       em vários conjuntos de dados de imagens de moedas.
       Pode ser adaptado para funcionar com outros modelos ou datasets, efetuando as mudanças necessárias.
"""
import os
import torch
import pandas as pd
from collections import Counter
from skimage import io
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models
import torch.nn as nn

# -------------------------------
# Configuração
# -------------------------------
MODEL_PATH = './mobilenet_v2-finetuned.pth'
BATCH_SIZE = 8

CSV_ROOT_PAIRS = [
    ('../imagens_treino/GR2004/Greece.csv', '../imagens_treino/GR2004/'),
    ('../imagens_treino/FR2007 treaty/France.csv', '../imagens_treino/FR2007 treaty/'),
    ('../imagens_treino/NL2013-2/Netherlands.csv', '../imagens_treino/NL2013-2/'),
    ('../imagens_treino/PT2016-1/Portugal.csv', '../imagens_treino/PT2016-1/'),
    ('../imagens_treino/PT2016-2/Portugal2.csv', '../imagens_treino/PT2016-2/'),
]

# -------------------------------
# Transformações
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
# Dataset
# -------------------------------
class CoinsDataset(Dataset):
    """
    @brief Um Dataset PyTorch personalizado para carregar imagens de moedas e os seus IDs.
    """
    def __init__(self, csv_file, root_dir, transform=None):
        """
        @brief Inicializa o CoinsDataset.
        @param csv_file Caminho para o ficheiro CSV contendo dados das imagens (IDs e nomes de ficheiro).
        @param root_dir Pasta onde os ficheiros de imagem estão localizados.
        @param transform Transformação opcional a ser aplicada numa amostra.
        """
        self.coins_data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        """
        @brief Retorna o número total de amostras no dataset.
        @return O número de amostras.
        """
        return len(self.coins_data)

    def __getitem__(self, idx):
        """
        @brief Recupera um item (imagem e o seu ID) do dataset.
        @param idx Índice do item a ser recuperado.
        @return Uma tupla contendo a imagem (Tensor) e o seu ID (int).
        """
        label = int(self.coins_data.iloc[idx, 0])
        img_path = os.path.join(self.root_dir, self.coins_data.iloc[idx, 1])
        img = io.imread(img_path)

        if img.shape[2] == 4: # Handle images with alpha channel
            img = img[:, :, :3]

        if self.transform:
            img = self.transform(img)

        return img, label

# -------------------------------
# Avaliação
# -------------------------------
def evaluate(model, loader, description="Avaliação"):
    """
    @brief Avalia o desempenho do modelo (precisão) num dado DataLoader.
    @param model O modelo PyTorch a ser avaliado.
    @param loader O DataLoader contendo os dados de avaliação.
    @param description Uma string que descreve o conjunto de avaliação para impressão.
    """
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

# -------------------------------
# Modelo
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carrega primeiro o modelo com o número correto de saídas
model = models.mobilenet_v2()
# Ajusta a ultima camada do modelo de forma a ter a mesma quantidade de classes do treino.
# O modelo original da ImageNet tem 1000 classes, por isso temos de alterar para 497
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 497)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)

# -------------------------------
# Avaliação dos CSVs
# -------------------------------
print("==> Avaliando conjuntos completos (sem divisão):")

for csv_path, root_path in CSV_ROOT_PAIRS:
    ds = CoinsDataset(csv_path, root_path, transform=data_transforms)

    labels = ds.coins_data.iloc[:, 0].values
    label_counts = Counter(labels)
    # Filtra as labels que aparecem menos de 2 vezes, ja que tambem nao iriam ser usadas no treino
    valid_indices = [i for i, label in enumerate(labels) if label_counts[label] >= 2]

    if not valid_indices:
        print(f"[Aviso] Nenhuma classe com ≥ 2 amostras em: {csv_path}")
        continue

    filtered_ds = Subset(ds, valid_indices)
    test_loader = DataLoader(filtered_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    label = os.path.splitext(os.path.basename(csv_path))[0]
    evaluate(model, test_loader, f"Teste - {label}")