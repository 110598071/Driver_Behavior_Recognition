import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import util
from datetime import datetime
import time
import os
import csv
import shutil
import feature_data
import math
import torch
import torchvision.models as models

# DRIVER_IMG = '../csv/driver_imgs_list.csv'
# IMG_DIR_PATH = "D:/Project/data/driver distraction/ggg/"
# TRAIN_IMG = "D:/Project/pytorch-openpose/experiment_1121/origin_img/train/"
# TEST_IMG = "D:/Project/pytorch-openpose/experiment_1121/origin_img/test/"

# driver_dict = {}

# with open(DRIVER_IMG, newline='') as driver_img:
#     driver_img_reader = csv.reader(driver_img)
#     for img in driver_img_reader:
#         if img[2] == 'img':
#             continue
#         driver_dict[img[2]] = (int)(img[0][2:4])
    
# for class_dir in os.listdir(IMG_DIR_PATH):
#     class_img = IMG_DIR_PATH + class_dir + "/"
#     for img in os.listdir(class_img):
#         img_path = class_img + img
#         if (driver_dict.get(img) < 51):
#             shutil.copy(img_path, TRAIN_IMG+class_dir+"/")
#         else:
#             shutil.copy(img_path, TEST_IMG+class_dir+"/")

FC_NUM_EPOCHS = 10
FC_LEARNING_RATE = 0.0001

WHOLE_NUM_EPOCHS = 20
WHOLE_LEARNING_RATE = 0.0003

BATCH_SIZE = 100
MOMENTUM = 0.9

resnet = models.resnet101(weights='IMAGENET1K_V1', progress=True)
optimizer = torch.optim.SGD(resnet.fc.parameters(), lr=FC_LEARNING_RATE, momentum=MOMENTUM)
fc_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=0, eta_min=0.000004)

print(fc_lr_scheduler.get_last_lr())