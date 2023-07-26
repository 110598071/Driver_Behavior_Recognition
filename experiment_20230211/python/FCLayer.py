import torch
import torch.nn as nn

class ResNet_fc(nn.Module):
    def __init__(self, needMoreLayers):
        super(ResNet_fc, self).__init__()
        self.needMoreLayers = needMoreLayers
        self.flatten = nn.Flatten()
        self.bn1 = nn.BatchNorm1d(2048)

        self.dropout1 = nn.Dropout(p=0.6)
        self.linear = nn.Linear(2048, 512)
        self.dropout_addition = nn.Dropout(p=0.25)
        self.bn2 = nn.BatchNorm1d(512)
        self.relu = nn.ReLU()

        self.dropout2 = nn.Dropout(p=0.6)
        self.output = nn.Linear(512, 10)
    
    def forward(self, x):
        if (self.needMoreLayers):
            # x = self.flatten(x)
            # x = self.bn1(x)
            # x = self.relu(x)

            x = self.dropout1(x)
            x = self.linear(x)
            x = self.relu(x)
            x = self.dropout_addition(x)
            x = self.bn2(x) 
            
        x = self.dropout2(x)
        x = self.output(x)
        return x
    
class ResNet_test_fc(nn.Module):
    def __init__(self):
        super(ResNet_test_fc, self).__init__()
        self.flatten = nn.Flatten()
        self.bn1 = nn.BatchNorm1d(4096)

        self.dropout1 = nn.Dropout1d(p=0.25)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm1d(4096)
        self.linear1 = nn.Linear(4096, 512)

        self.dropout2 = nn.Dropout1d(p=0.5)
        self.linear2 = nn.Linear(512, 10)
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.bn1(x)
        x = self.dropout1(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.linear1(x)
        x = self.dropout2(x)
        x = self.linear2(x)
        return x
    
class AlexNet_classifier(nn.Module):
    def __init__(self):
        super(AlexNet_classifier, self).__init__()

        self.dropout1 = nn.Dropout(p=0.5, inplace=False)

        self.linear1 = nn.Linear(9216, 4096, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(p=0.5, inplace=False)
        self.bn1 = nn.BatchNorm1d(4096)

        self.linear2 = nn.Linear(4096, 1024, bias=True)
        self.dropout3 = nn.Dropout(p=0.5, inplace=False)
        self.bn2 = nn.BatchNorm1d(1024)

        self.output = nn.Linear(1024, 10, bias=True)
    
    def forward(self, x):
        x = self.dropout1(x)

        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.bn1(x)

        x = self.linear2(x)
        x = self.relu(x)
        x = self.dropout3(x)
        x = self.bn2(x)

        x = self.output(x)
        return x
    
class EfficientNet_classifier(nn.Module):
    def __init__(self):
        super(EfficientNet_classifier, self).__init__()

        self.dropout1 = nn.Dropout(p=0.7, inplace=False)
        self.linear1 = nn.Linear(1280, 640, bias=True)
        self.relu = nn.ReLU(inplace=True)

        self.dropout2 = nn.Dropout(p=0.6, inplace=False)
        self.bn = nn.BatchNorm1d(640)
        self.output = nn.Linear(640, 10, bias=True)
    
    def forward(self, x):
        x = self.dropout1(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.bn(x)
        x = self.output(x)
        return x
    
class MobileNet_classifier(nn.Module):
    def __init__(self):
        super(MobileNet_classifier, self).__init__()

        self.dropout1 = nn.Dropout(p=0.5, inplace=False)
        self.linear1 = nn.Linear(960, 480, bias=True)
        self.hardswish = nn.Hardswish(inplace=True)

        self.dropout2 = nn.Dropout(p=0.25, inplace=False)
        self.bn = nn.BatchNorm1d(480)
        self.output = nn.Linear(480, 10, bias=True)
    
    def forward(self, x):
        x = self.dropout1(x)
        x = self.linear1(x)
        x = self.hardswish(x)
        x = self.dropout2(x)
        x = self.bn(x)
        x = self.output(x)
        return x
    
class InceptionV3_fc(nn.Module):
    def __init__(self):
        super(InceptionV3_fc, self).__init__()
        
        self.linear = nn.Linear(2048, 512)
        self.relu = nn.ReLU(inplace=True)
        self.dp = nn.Dropout(p=0.6)
        self.bn = nn.BatchNorm1d(512)
        
        self.output = nn.Linear(512, 10, bias=True)
    
    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.dp(x)
        x = self.bn(x)
        
        x = self.output(x)
        return x
    
class VGG16_classifier(nn.Module):
    def __init__(self):
        super(VGG16_classifier, self).__init__()
        self.linear1 = nn.Linear(25088, 4096)
        self.relu = nn.ReLU(inplace=True)
        self.dp1 = nn.Dropout(p=0.5)
        self.bn1 = nn.BatchNorm1d(4096)

        self.linear2 = nn.Linear(4096, 512)
        self.dp2 = nn.Dropout(p=0.5)
        self.bn2 = nn.BatchNorm1d(512)
        
        self.output = nn.Linear(512, 10, bias=True)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dp1(x)
        x = self.bn1(x)

        x = self.linear2(x)
        x = self.relu(x)
        x = self.dp2(x)
        x = self.bn2(x)
        
        x = self.output(x)
        return x
    
class Proposed_CNN(nn.Module):
    def __init__(self):
        super(Proposed_CNN, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        self.conv2d = nn.Conv2d(3, 32, kernel_size=(3, 3), padding=1)
        self.bn = nn.BatchNorm2d(32)
        self.max_pooling2d = nn.MaxPool2d(kernel_size=2) 

        self.conv2d_1 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.bn_1 = nn.BatchNorm2d(64)
        self.conv2d_2 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1)
        self.bn_2 = nn.BatchNorm2d(64)
        self.max_pooling2d = nn.MaxPool2d(kernel_size=2) 

        self.conv2d_3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.bn_3 = nn.BatchNorm2d(128)

        self.flatten = nn.Flatten()
        self.dense = nn.Linear(100352, 128)
        self.dense_1 = nn.Linear(128, 128)
        self.dense_2 = nn.Linear(128, 64)
        self.dropout = nn.Dropout()
        self.dense_3 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = self.conv2d(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.max_pooling2d(x)

        x = self.conv2d_1(x)
        x = self.bn_1(x)
        x = self.relu(x)
        x = self.conv2d_2(x)
        x = self.bn_2(x)
        x = self.relu(x)
        x = self.max_pooling2d(x)

        x = self.conv2d_3(x)
        x = self.bn_3(x)
        x = self.relu(x)

        x = self.flatten(x)
        x = self.dense(x)
        x = self.relu(x)
        x = self.dense_1(x)
        x = self.relu(x)
        x = self.dense_2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense_3(x)

        return x
    
class DecreasingFilterCNN(nn.Module):
    def __init__(self):
        super(DecreasingFilterCNN, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        self.conv2d_1 = nn.Conv2d(3, 32, kernel_size=(11, 11), padding=5)
        self.bn_1 = nn.BatchNorm2d(32)
        self.max_pooling2d_1 = nn.MaxPool2d(kernel_size=2)

        self.conv2d_2 = nn.Conv2d(32, 64, kernel_size=(9, 9), padding=4)
        self.bn_2 = nn.BatchNorm2d(64)
        self.max_pooling2d_2 = nn.MaxPool2d(kernel_size=2)

        self.conv2d_3 = nn.Conv2d(64, 128, kernel_size=(5, 5), padding=2)
        self.bn_3 = nn.BatchNorm2d(128)
        self.max_pooling2d_3 = nn.MaxPool2d(kernel_size=2)

        self.conv2d_4 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1)
        self.bn_4 = nn.BatchNorm2d(256)
        self.max_pooling2d_4 = nn.MaxPool2d(kernel_size=2)

        self.dropout = nn.Dropout2d(p=0.5)
        self.global_average_pooling = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.flatten = nn.Flatten()
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv2d_1(x)
        x = self.relu(x)
        x = self.bn_1(x)
        x = self.max_pooling2d_1(x)

        x = self.conv2d_2(x)
        x = self.relu(x)
        x = self.bn_2(x)
        x = self.max_pooling2d_2(x)

        x = self.conv2d_3(x)
        x = self.relu(x)
        x = self.bn_3(x)
        x = self.max_pooling2d_3(x)

        x = self.conv2d_4(x)
        x = self.relu(x)
        x = self.bn_4(x)
        x = self.max_pooling2d_4(x)

        x = self.dropout(x)
        x = self.global_average_pooling(x)
        x = self.flatten(x)
        x = self.output(x)
        return x

class SLP(nn.Module):
    def __init__(self):
        super(SLP, self).__init__()
        INPUT_DIM = 43

        self.relu = nn.ReLU(inplace=True)
        self.dp = nn.Dropout(p=0.2)
        self.bn = nn.BatchNorm1d(INPUT_DIM)
        self.output = nn.Linear(INPUT_DIM, 10)

    def forward(self, x):
        x = self.dp(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.output(x)
        return x
    
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        INPUT_DIM = 43

        self.relu = nn.ReLU(inplace=True)
        self.dp1 = nn.Dropout(p=0.2)
        self.bn1 = nn.BatchNorm1d(INPUT_DIM)
        self.linear = nn.Linear(INPUT_DIM, 30)

        self.dp2 = nn.Dropout(p=0.2)
        self.bn2 = nn.BatchNorm1d(30)
        self.output = nn.Linear(30, 10)

    def forward(self, x):
        x = self.dp1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.linear(x)

        x = self.dp2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.output(x)
        return x
    
class Swin_Transformer_tiny_head(nn.Module):
    def __init__(self):
        super(Swin_Transformer_tiny_head, self).__init__()
        self.linear = nn.Linear(768, 512)
        self.relu = nn.ReLU(inplace=True)
        self.dp = nn.Dropout(p=0.5)
        self.bn = nn.BatchNorm1d(512)
        
        self.output = nn.Linear(512, 10, bias=True)
    
    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.dp(x)
        x = self.bn(x)
        
        x = self.output(x)
        return x

class EnsembleModel(nn.Module):
    def __init__(self, modelA, modelB):
        super(EnsembleModel, self).__init__()
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU(inplace=True)
        self.modelA = modelA
        self.modelB = modelB

        self.modelA.classifier = nn.Identity()
        self.modelB.fc = nn.Identity()
        self.dropout = nn.Dropout(p=0.5)
        self.classifier = nn.Linear(11264, 10)

    def forward(self, x):
        x1 = self.modelA(x.clone())
        x1 = x1.view(x1.size(0), -1)
        x2 = self.modelB(x)
        x2 = x2.view(x2.size(0), -1)
        x = torch.cat((x1, x2), dim=1)

        x = self.dropout(x)
        x = self.classifier(self.relu(x))
        return x