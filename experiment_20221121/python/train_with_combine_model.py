from sklearn.ensemble import RandomForestClassifier
import torchvision.models as models
from torchvision import datasets, models, transforms
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from collections import OrderedDict
import feature_data
import train_with_pytorch
import pytorch_util
import os

FC_NUM_EPOCHS = 10
FC_LEARNING_RATE = 0.000004

WHOLE_NUM_EPOCHS = 20
WHOLE_LEARNING_RATE = 0.000001

BATCH_SIZE = 32
MOMENTUM = 0.9

class FCLayer(nn.Module):
    def __init__(self):
        super(FCLayer, self).__init__()
        self.flatten = nn.Flatten()
        
        self.bn = nn.BatchNorm1d(2048)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.25)
        self.linear = nn.Linear(2048, 512)

        self.dropout2 = nn.Dropout(p=0.5)
        self.output = nn.Linear(512, 10) 
    
    def forward(self, x):
        x = self.flatten(x)
        
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.linear(x)

        x = self.dropout2(x)
        x = self.output(x)
        return x

def train_with_skeleton_data(data_train, label_train):
    clf = RandomForestClassifier()
    clf.fit(data_train, label_train)
    return clf

def predict_with_skeleton_data(clf, data):
    return clf.predict_proba(data)[0]

if __name__ == '__main__':
    # (data_train, label_train), (data_test, label_test) = feature_data.get_train_and_test_data()
    # clf = train_with_skeleton_data(data_train, label_train)

    # for data in data_test:
    #     print(predict_with_skeleton_data(clf, [data]))

    #========================================#

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize([360, 360]),
            transforms.ToTensor(),
        ]),
        'val': transforms.Compose([
            transforms.Resize([360, 360]),
            transforms.ToTensor(),
        ]),
    }

    data_dir = 'D:/Project/pytorch-openpose/experiment_1121/origin_img'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=4) for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resnet = models.resnet101(weights='IMAGENET1K_V1', progress=True)

    for param in resnet.parameters():
        param.requires_grad = False
    resnet.fc = FCLayer()
    resnet = resnet.to(device)

    # print(resnet)
    criterion = nn.CrossEntropyLoss()
    fc_optimizer = torch.optim.Adam(resnet.fc.parameters(), lr=FC_LEARNING_RATE, momentum=MOMENTUM)
    fc_lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(fc_optimizer, max_lr=0.0001, total_steps=FC_NUM_EPOCHS , verbose=True)
    resnet_fc_train, best_acc = pytorch_util.train_model(resnet, criterion, fc_optimizer, fc_lr_scheduler, dataloaders, device, dataset_sizes, num_epochs=FC_NUM_EPOCHS)
    pytorch_util.save_model(resnet_fc_train, best_acc)

    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()

    for param in resnet_fc_train.parameters():
        param.requires_grad = True
    whole_optimizer = torch.optim.SGD(resnet.fc.parameters(), lr=FC_LEARNING_RATE, momentum=MOMENTUM)
    whole_lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(whole_optimizer, max_lr=0.000004, total_steps=WHOLE_NUM_EPOCHS, verbose=True)
    resnet_whole_train, best_acc = pytorch_util.train_model(resnet, criterion, whole_optimizer, whole_lr_scheduler, dataloaders, device, dataset_sizes, num_epochs=WHOLE_NUM_EPOCHS)
    pytorch_util.save_model(resnet_whole_train, best_acc)