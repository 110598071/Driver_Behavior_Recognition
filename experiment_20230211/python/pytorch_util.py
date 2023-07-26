import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import time
import copy
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import random
from datetime import datetime

class EarlyStopping():
    def __init__(self, patient=10):
        self.patient = patient
        self.quota = patient
        self.previousLoss = 1000.0

    def check(self, loss):
        if loss < self.previousLoss:
            self.previousLoss = loss
            if self.quota == self.patient:
                return 0  # better val loss
            else:
                self.quota = self.patient
                return 1  # renew quota
        else:
            self.quota -= 1
            # self.previousLoss = loss
            if self.quota == 0:
                return 3  # early stop
            else:
                return 2  # quota -1

    def print(self):
        print("quota: {}, previousLoss: {:.4f}".format(self.quota, self.previousLoss))
        print()


def get_data_transforms():
    RESIZE = [400 for _ in range(2)]
    data_transforms = {
        'train': transforms.Compose([
            # transforms.Grayscale(),
            transforms.Resize(RESIZE),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            # transforms.Lambda(lambda x: x.repeat(3,1,1)),
            transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5)),
        ]),
        'test': transforms.Compose([
            # transforms.Grayscale(),
            transforms.Resize(RESIZE),
            transforms.ToTensor(),
            # transforms.Lambda(lambda x: x.repeat(3,1,1)),
            transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5)),
        ]),
        'augmentation': transforms.Compose([
            transforms.Resize(RESIZE),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            # transforms.RandomHorizontalFlip(p=0.2),
            transforms.RandomRotation(20),
            # transforms.RandAugment(),
            transforms.ToTensor(),
            transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5)),
        ]),
        'camera2_augmentation': transforms.Compose([
            transforms.Resize(RESIZE),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            transforms.RandomRotation(20),
            # transforms.RandAugment(),
            transforms.ToTensor(),
            transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5)),
        ]),
    }
    return data_transforms

def train_dataset_augmentation(datasetType, data_dir):
    if (datasetType == 'train'):
        combine_datasets = [datasets.ImageFolder(os.path.join(data_dir, 'train'), get_data_transforms()['train'])]
        for i in range(2):
            combine_datasets.append(datasets.ImageFolder(os.path.join(data_dir, 'train'), get_data_transforms()['augmentation']))

        image_datasets = torch.utils.data.ConcatDataset(combine_datasets)
    else:
        image_datasets = datasets.ImageFolder(os.path.join(data_dir, datasetType), get_data_transforms()[datasetType])
    return image_datasets

def test_image(model, modelPath, imgPathList):
    checkpoint = torch.load(modelPath)
    model.load_state_dict(checkpoint)
    model.eval()

    output_list = []
    pbar = tqdm(imgPathList)
    for imgPath in pbar:
        img = Image.open(imgPath)
        imgInput = get_data_transforms()['test'](img)
        imgInput = imgInput.unsqueeze(0)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        imgInput = imgInput.to(device)

        output = model(imgInput)
        output_cpu = output.cpu().detach().numpy().tolist()[0]
        output_list.append([float(i) for i in output_cpu])
        pbar.set_description('CNN Inference')
    return output_list

def setup_seed(random_seed):
    torch.cuda.empty_cache()
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    np.random.seed(random_seed)
    random.seed(random_seed)

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
