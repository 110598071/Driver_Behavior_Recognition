import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import FCLayer

def get_pretrained_ResNet(layers):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if (layers == "101"):
        resnet = models.resnet101(weights='IMAGENET1K_V1', progress=True)
        needMoreLayers = True
    elif (layers == "50"):
        resnet = models.resnet50(weights='IMAGENET1K_V1', progress=True)
        needMoreLayers = True
    elif (layers == "34"):
        resnet = models.resnet34(weights='IMAGENET1K_V1', progress=True)
        needMoreLayers = False
    elif (layers == "18"):
        resnet = models.resnet18(weights='IMAGENET1K_V1', progress=True)
        needMoreLayers = False
    elif (layers == "152"):
        resnet = models.resnet152(weights='IMAGENET1K_V1', progress=True)
        needMoreLayers = True
    
    for param in resnet.parameters():
        param.requires_grad = False
    resnet.fc = FCLayer.ResNet_fc(needMoreLayers)
    resnet = resnet.to(device)
    return resnet

def get_original_ResNet(layers):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if (layers == "101"):
        resnet = models.resnet101(progress=True)
        needMoreLayers = True
    elif (layers == "50"):
        resnet = models.resnet50(progress=True)
        needMoreLayers = True
    elif (layers == "34"):
        resnet = models.resnet34(progress=True)
        needMoreLayers = False
    elif (layers == "18"):
        resnet = models.resnet18(progress=True)
        needMoreLayers = False

    resnet.fc = FCLayer.ResNet_fc(needMoreLayers)
    for param in resnet.parameters():
        param.requires_grad = True
    resnet = resnet.to(device)
    return resnet

def get_test_ResNet():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    resnet = models.resnet101(weights='IMAGENET1K_V1', progress=True)
    for param in resnet.parameters():
        param.requires_grad = False
    resnet.avgpool = nn.AdaptiveAvgPool2d(output_size=(2, 1))
    resnet.fc = FCLayer.ResNet_test_fc()
    resnet = resnet.to(device)
    return resnet

def get_pretrained_AlexNet():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    alexnet = models.alexnet(weights='IMAGENET1K_V1', progress=True)
    for param in alexnet.parameters():
        param.requires_grad = False
    alexnet.classifier = FCLayer.AlexNet_classifier()
    alexnet = alexnet.to(device)
    return alexnet

def get_original_AlexNet():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    alexnet = models.alexnet(progress=True)
    alexnet.classifier = FCLayer.AlexNet_classifier()
    for param in alexnet.parameters():
        param.requires_grad = True
    alexnet = alexnet.to(device)
    return alexnet

def get_pretrained_EfficientNet_s():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    efficientnet = models.efficientnet_v2_s(progress=True)
    # checkpoint = torch.load("../efficientnet_v2_s-dd5fe13b.pth")
    # efficientnet.load_state_dict(checkpoint)
    for param in efficientnet.parameters():
        param.requires_grad = False
    efficientnet.classifier = FCLayer.EfficientNet_classifier()
    efficientnet = efficientnet.to(device)
    return efficientnet

def get_pretrained_EfficientNet_m():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    efficientnet = models.efficientnet_v2_m(progress=True)
    # checkpoint = torch.load("../efficientnet_v2_m-dc08266a.pth")
    # efficientnet.load_state_dict(checkpoint)
    for param in efficientnet.parameters():
        param.requires_grad = False
    efficientnet.classifier = FCLayer.EfficientNet_classifier()
    efficientnet = efficientnet.to(device)
    return efficientnet

def get_pretrained_Swin_Transformer_small():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    swin_transformer_small = models.swin_s(progress=True)
    # checkpoint = torch.load("../swin_s-5e29d889.pth")
    # swin_transformer_small.load_state_dict(checkpoint)
    for param in swin_transformer_small.parameters():
        param.requires_grad = False
    swin_transformer_small.head = FCLayer.Swin_Transformer_tiny_head()
    swin_transformer_small = swin_transformer_small.to(device)
    return swin_transformer_small

def get_pretrained_MobileNet():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mobilenet = models.mobilenet_v3_large(weights='IMAGENET1K_V1', progress=True)
    for param in mobilenet.parameters():
        param.requires_grad = False
    mobilenet.classifier = FCLayer.MobileNet_classifier()
    mobilenet = mobilenet.to(device)
    return mobilenet

def get_pretrained_InceptionV3():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    inception_v3 = models.inception_v3(weights='IMAGENET1K_V1', progress=True)
    for param in inception_v3.parameters():
        param.requires_grad = False
    # inception_v3.dropout = nn.Identity()
    inception_v3.fc = FCLayer.InceptionV3_fc()
    inception_v3 = inception_v3.to(device)
    return inception_v3

def get_pretrained_VGG16():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vgg16 = models.vgg16(weights='IMAGENET1K_V1', progress=True)
    for param in vgg16.parameters():
        param.requires_grad = False
    vgg16.classifier = FCLayer.VGG16_classifier()
    vgg16 = vgg16.to(device)
    return vgg16

def get_proposed_cnn():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    proposed_cnn = FCLayer.Proposed_CNN()
    proposed_cnn = proposed_cnn.to(device)
    return proposed_cnn

def get_decreasing_filter_cnn():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    proposed_cnn = FCLayer.DecreasingFilterCNN()
    proposed_cnn = proposed_cnn.to(device)
    return proposed_cnn

def get_SLP():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    SLP = FCLayer.SLP()
    SLP = SLP.to(device)
    return SLP

def get_MLP():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    MLP = FCLayer.MLP()
    MLP = MLP.to(device)
    return MLP