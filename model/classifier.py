import torch
import torch.nn as nn
from typing import Tuple, List
from torchvision import models, transforms
from PIL import Image
from config import CLASS_LABELS, MODEL_PATH
import torch.nn.functional as F


def get_model():
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASS_LABELS))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

def get_model_by_name(model_path: str, num_classes: int):
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    
    model.eval()
    return model


def predict(image: Image.Image, model, class_labels: List[str] = None) -> Tuple[str, float]:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, pred = torch.max(probabilities, dim=1)
    print(pred)

    if class_labels is None:
        class_labels = CLASS_LABELS

    return class_labels[pred.item()], confidence.item()

