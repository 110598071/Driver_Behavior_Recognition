import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import FCLayer

def get_original_AlexNet():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    alexnet = models.alexnet(progress=True)
    alexnet.classifier = FCLayer.AlexNet_classifier()
    for param in alexnet.parameters():
        param.requires_grad = True
    alexnet = alexnet.to(device)
    return alexnet