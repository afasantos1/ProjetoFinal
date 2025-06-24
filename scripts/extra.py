import torch
from torchvision import models
import torch.nn as nn

# Load base MobileNetV2 (no pretrained weights, because you're loading your own)
model = models.mobilenet_v2(pretrained=False)

# Modify the classifier to match the fine-tuned model (497 classes)
model.classifier[1] = nn.Linear(model.last_channel, 497)

# Load fine-tuned weights (trained on 497 classes)
model.load_state_dict(torch.load("mobilenet_v2-finetuned.pth", map_location=torch.device('cpu')))
model.eval()

# Convert to TorchScript for PyTorch Mobile
scripted_model = torch.jit.script(model)
scripted_model.save("model.pt")
