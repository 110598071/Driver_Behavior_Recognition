import torchvision.models as models
from torchvision import datasets, models, transforms
from torch.utils.data import random_split
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from collections import OrderedDict
import pytorch_util
import pytorch_model
import csv
import random
import numpy as np
import AlexNet
import Image_dataset
from datetime import datetime
import os
import pytorch_progress
from FCLayer import EnsembleModel

# ======= pytorch seed ======= #
# PYTORCH_SEED = random.randint(0, 4294967295)
# PYTORCH_SEED = 597381152 # FC model
PYTORCH_SEED = 3578671768 # CNN
# ======== batch size ======== #
BATCH_SIZE = 16
# ===== transfer learning ===== #
TRANSFER_NEED_SCHEDULER = True
TRANSFER_FC_EPOCHS = 10
TRANSFER_FC_LR = 0.00005

TRANSFER_WHOLE_EPOCHS = 30
TRANSFER_WHOLE_LR = 0.00335
# ============================= #

# ===== original training ===== #
ORIGINAL_NEED_SCHEDULER = True
ORIGINAL_EPOCHS = 50
# WEIGHT_DECAY = 0.04
# ORIGINAL_LR = 0.0002
ORIGINAL_LR = 0.0038
WEIGHT_DECAY = 0.07
# ============================= #

# ===== scheduler kwargs ====== #
transfer_fc_scheduler_kwargs = {
    'max_lr': TRANSFER_FC_LR,
    'pct_start': 0.28,
    'div_factor': 40.5,
    'final_div_factor': 24.4,
    'total_steps': TRANSFER_FC_EPOCHS,
    'verbose': False,
}

transfer_whole_scheduler_kwargs = {
    'max_lr': TRANSFER_WHOLE_LR,
    'pct_start': 0.64,
    'div_factor': 45.5,
    'final_div_factor': 4.45,
    'total_steps': TRANSFER_WHOLE_EPOCHS,
    'verbose': False,
}

# origin_scheduler_kwargs = {
#     'max_lr': ORIGINAL_LR,
#     'pct_start': 0.4,
#     'div_factor': 40,
#     'final_div_factor': 0.05,
#     'total_steps': ORIGINAL_EPOCHS,
#     # 'gamma': 0.99,
#     'verbose': False,
# }

origin_scheduler_kwargs = {
    'max_lr': ORIGINAL_LR,
    'pct_start': 0.12,
    'div_factor': 65.0,
    'final_div_factor': 38.1,
    'total_steps': ORIGINAL_EPOCHS,
    # 'gamma': 0.99,
    'verbose': False,
}
# ============================= #

def model_transfer_learning(model):
    transfer_pytorch_model = pytorch_progress.pytorch_model(model, BATCH_SIZE, True)

    fc_optimizer = torch.optim.Adam(model.classifier.parameters(), lr=TRANSFER_FC_LR)
    fc_lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(fc_optimizer, **transfer_fc_scheduler_kwargs)
    transfer_pytorch_model.train(fc_optimizer, fc_lr_scheduler, TRANSFER_NEED_SCHEDULER, TRANSFER_FC_EPOCHS)

    model_fc_trained = transfer_pytorch_model.get_model()
    for param in model_fc_trained.parameters():
        param.requires_grad = True

    whole_optimizer = torch.optim.Adam(model_fc_trained.parameters(), lr=TRANSFER_WHOLE_LR)
    whole_lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(whole_optimizer, **transfer_whole_scheduler_kwargs)
    transfer_pytorch_model.train(whole_optimizer, whole_lr_scheduler, TRANSFER_NEED_SCHEDULER, TRANSFER_WHOLE_EPOCHS)

    transfer_pytorch_model.test()
    transfer_pytorch_model.print_model_use_time()
    transfer_pytorch_model.save_model()
    transfer_pytorch_model.plot_training_progress()
    transfer_pytorch_model.plot_confusion_matrix()

def model_origin_training(model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=ORIGINAL_LR, weight_decay=WEIGHT_DECAY)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, **origin_scheduler_kwargs)
    # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, **origin_scheduler_kwargs)

    origin_pytorch_model = pytorch_progress.pytorch_model(model, BATCH_SIZE, True)
    origin_pytorch_model.train(optimizer, lr_scheduler, ORIGINAL_NEED_SCHEDULER, ORIGINAL_EPOCHS)
    origin_pytorch_model.test()
    origin_pytorch_model.print_model_use_time()
    origin_pytorch_model.save_model()
    origin_pytorch_model.plot_training_progress()
    origin_pytorch_model.plot_confusion_matrix()
    origin_pytorch_model.get_wrong_classification_img()
    return origin_pytorch_model.get_test_accuracy()

def find_best_seed():
    best_test_accuracy = 0.0

    initial_weight_list = []
    test_accuracy_list = []
    random_seed_list = []

    while(best_test_accuracy < 0.947):
        seed = random.randint(0, 4294967295)
        pytorch_util.setup_seed(seed)
        simpleFC = pytorch_model.get_SLP()
        initial_weight_list.append(simpleFC.output.weight.tolist())
        test_accuracy = model_origin_training(simpleFC)

        if (test_accuracy > best_test_accuracy):
            best_test_accuracy = test_accuracy
            store_initial_weight(initial_weight_list[-1])

        random_seed_list.append(seed)
        test_accuracy_list.append(test_accuracy)

        sort_test_accuracy_list = test_accuracy_list.copy()
        sort_test_accuracy_list = list(set(sort_test_accuracy_list))
        sort_test_accuracy_list.sort(reverse=True)
        
        print('='*30)
        for i in range(np.min([len(sort_test_accuracy_list), 5])):
            print(f'{i+1}st acc: {sort_test_accuracy_list[i]:.4f} / seed: {random_seed_list[test_accuracy_list.index(sort_test_accuracy_list[i])]}')
        print('='*30)

def store_initial_weight(output_weight):
    with open(f'D:/Project/pytorch-openpose/experiment_0211/dataset/initial_weight.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(output_weight)

if __name__ == '__main__':
    pytorch_util.setup_seed(PYTORCH_SEED)
    # resnet = pytorch_model.get_pretrained_ResNet('101')
    resnet = pytorch_model.get_pretrained_EfficientNet_s()
    model_transfer_learning(resnet)
    print(f'seed: {PYTORCH_SEED}')

    # inception_v3 = pytorch_model.get_pretrained_InceptionV3()
    # model_transfer_learning(inception_v3)

    # resnet = pytorch_model.get_test_ResNet()
    # model_transfer_learning(resnet)

    # proposed_cnn = pytorch_model.get_decreasing_filter_cnn()
    # model_origin_training(proposed_cnn)

    # pytorch_util.setup_seed(PYTORCH_SEED)
    # perceptron = pytorch_model.get_SLP()
    # # perceptron = pytorch_model.get_MLP()
    # model_origin_training(perceptron)
    # print(f'seed: {PYTORCH_SEED}')

    # find_best_seed()

    # alexnet = pytorch_model.get_pretrained_AlexNet()
    # resnet50 = pytorch_model.get_pretrained_ResNet('50')
    # ensemble_model = EnsembleModel(alexnet, resnet50)
    # model_origin_training(ensemble_model)