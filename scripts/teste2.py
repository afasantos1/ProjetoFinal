import torch
from torchvision import transforms
from PIL import Image

# Load the TorchScript model
model = torch.jit.load("model.pt")
model.eval()

# Load and preprocess image
img = Image.open("../imagens_treino/PT2016-2/IMG_20250619_190510.png").convert("RGB")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # TORCHVISION_NORM_MEAN_RGB
        std=[0.229, 0.224, 0.225]    # TORCHVISION_NORM_STD_RGB
    )
])
input_tensor = transform(img).unsqueeze(0)  # Shape: [1, 3, 224, 224]

# Run inference
with torch.no_grad():
    output = model(input_tensor)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    predicted_class = probabilities.argmax().item()

print(f"Predicted class index: {predicted_class}")
