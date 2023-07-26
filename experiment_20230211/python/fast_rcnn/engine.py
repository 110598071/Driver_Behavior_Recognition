from .config import DEVICE, NUM_CLASSES, NUM_EPOCHS, OUT_MODEL_DIR, OUT_LOSS_DIR, RESIZE_TO, BATCH_SIZE, CLASSES, SEMI_SUPERVISED_ACCEPTANCE, WEIGHT_DECAY, LEARNING_RATE, SCHEDULER_KWARGS, VALID_DIR
from .IoU_computation import IoU_computation
from .mAP_computation import mAP_inference, get_ground_true_boxes, compute_mAP
from .model import create_model
from .utils import Averager, EarlyStopping
from .inference import inference
from .datasets import train_loader, valid_loader, InferenceDataset, get_train_transform, train_dataset, collate_fn
from datetime import datetime
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset
import os
import torch
import torchvision
import matplotlib.pyplot as plt
import time
import copy
import glob

class engine:
    def __init__(self, train_loader, valid_loader, iter):
        self.model = create_model(num_classes=NUM_CLASSES)
        self.model = self.model.to(DEVICE)

        self.test_images = []
        for subdir in os.listdir(VALID_DIR):
            self.test_images.extend(glob.glob(f"{VALID_DIR}/{subdir}/*.jpg"))

        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)
        # self.optimizer = torch.optim.AdamW(params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, **SCHEDULER_KWARGS)

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.iter = iter

        self.early_stopping = EarlyStopping()
        self.train_loss_hist = Averager()
        self.val_loss_hist = Averager()

        self.train_loss_list = []
        self.val_loss_list = []

        self.best_model_wts = copy.deepcopy(self.model.state_dict())
        self.best_epoch = 1
        self.best_train_loss_value = 0.0
        self.best_val_loss_value = 100.0

        self.since = time.time()
        self.current_time = datetime.now().strftime("_%Y%m%d_%H%M")

    def train(self):
        for epoch in range(NUM_EPOCHS):
            print('='*30)
            print()
            print(f"ITER {self.iter}: EPOCH {epoch+1} of {NUM_EPOCHS}  #learning rate: {self.lr_scheduler.get_last_lr()[0]:.2e}")

            self.train_loss_hist.reset()
            self.val_loss_hist.reset()

            self.phase_progress(self.train_loader, 'train')
            self.phase_progress(self.valid_loader, 'val')
            self.train_loss_list.append(np.round(self.train_loss_hist.value, 3))
            self.val_loss_list.append(np.round(self.val_loss_hist.value, 3))

            print()
            print(f"train loss: {self.train_loss_hist.value:.3f}")   
            print(f"validation loss: {self.val_loss_hist.value:.3f}")

            if (self.val_loss_hist.value < self.best_val_loss_value):
                self.best_epoch = epoch+1
                self.best_train_loss_value = np.round(self.train_loss_hist.value, 3)
                self.best_val_loss_value = np.round(self.val_loss_hist.value, 3)
                self.best_model_wts = copy.deepcopy(self.model.state_dict())
            
            if self.early_stopping.check(self.val_loss_hist.value) == 3: break
        self.model.load_state_dict(self.best_model_wts)

    def phase_progress(self, data_loader, phase):
        # if phase == 'train':
        #     self.model.train()
        # else:
        #     self.model.eval()

        prog_bar = tqdm(data_loader, total=len(data_loader))
        for i, data in enumerate(prog_bar):
            self.optimizer.zero_grad()
            images, targets = data
            
            images = list(image.to(DEVICE) for image in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            with torch.set_grad_enabled(phase == 'train'):
                loss_dict = self.model(images, targets)

            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()

            if phase == 'train':
                self.train_loss_hist.send(loss_value)
                losses.backward()
                self.optimizer.step()
            else:
                self.val_loss_hist.send(loss_value)
            prog_bar.set_description(f'{phase} progress')
        if phase == 'train':
            self.lr_scheduler.step()

    def save_loss_plot_and_model(self):
        plt.style.use('ggplot')
        xtick = [i + 1 for i in range(len(self.train_loss_list))]
        plt.figure(figsize=(12, 8), dpi=100, linewidth=2)
        plt.plot(xtick, self.train_loss_list, 's-', color='r', label="train")
        plt.plot(xtick, self.val_loss_list, 'o-', color='g', label="validate")

        plt.title('Fast RCNN Loss')
        plt.xlabel("epochs")
        plt.ylabel('Loss')

        plt.legend(loc="best")
        plt.savefig(f"{OUT_LOSS_DIR}/Loss{self.current_time}.png")

        torch.save(self.model.state_dict(), f"{OUT_MODEL_DIR}/model{self.current_time}.pth")

    def compute_IoU(self):
        self.model.eval()
        iter_IoU_dict, best_detection_threshold = IoU_computation(self.model)
        print('='*30)
        print(f'best_detection_threshold: {best_detection_threshold}')
        print()
        for key, value in iter_IoU_dict.items():
            print(f'{key}: {value}')
        return iter_IoU_dict['average']
    
    def compute_mAP(self):
        print('='*30)
        self.model.eval()
        pred_boxes_list, pred_cls_list, pred_scores_list = mAP_inference(self.model, self.test_images)
        ground_true_boxes_list, ground_true_cls_list = get_ground_true_boxes(self.test_images)
        mAP, precision_list, recall_list, cls_list, AP_list = compute_mAP(pred_boxes_list, pred_cls_list, pred_scores_list, ground_true_boxes_list, ground_true_cls_list)
        
        print('='*30)
        print(f'mAP: {mAP}')
        print(f'avg Precision: {np.round(np.mean(precision_list), 4)}')
        print(f'avg Recall: {np.round(np.mean(recall_list), 4)}')
        for idx, cls in enumerate(cls_list):
            print(f'{("【"+cls+"】"):18s} AP: {AP_list[idx]:.4f} / precision: {precision_list[idx]:.3f} / recall: {recall_list[idx]:.3f}')
        return mAP

    def print_model_info(self):
        print('='*30)
        time_elapsed = time.time() - self.since
        print(f'model{self.current_time}')
        print(f'Train model complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print()
        print(f'occur epoch: {self.best_epoch}')
        print(f'best train loss: {self.best_train_loss_value}')
        print(f'best validation loss: {self.best_val_loss_value}')
        print('='*30)

    def get_best_loss(self):
        return self.best_val_loss_value
    
    def get_model(self):
        return self.model

def simple_RCNN_training():
    Engine = engine(train_loader, valid_loader, 0)
    Engine.train()
    Engine.save_loss_plot_and_model()
    Engine.compute_IoU()
    Engine.compute_mAP()
    Engine.print_model_info()

def semi_Supervised_Learning():
    since = time.time()
    current_iter = 0
    Engine = engine(train_loader, valid_loader, current_iter)
    Engine.train()
    best_Engine = Engine

    IMG_PATH = "D:/Project/pytorch-openpose/experiment_0211/dataset/object_detection_test/semi_supervised"
    imgPathList = []
    for cls in range(10):
        imgPathList.extend(glob.glob(f"{IMG_PATH}/c{cls}/*.jpg"))

    last_best_mAP = 0.0
    current_best_mAP = Engine.compute_mAP()
    best_mAP_history = [current_best_mAP]
    while (current_best_mAP >= last_best_mAP-SEMI_SUPERVISED_ACCEPTANCE):
        if current_best_mAP > last_best_mAP:
            best_Engine = Engine
            last_best_mAP = current_best_mAP
        model = Engine.get_model()
        model.eval()
        print('='*30)
        boxes_list, pred_cls_list, _ = inference(model, imgPathList, False)
        inferenceDataset = InferenceDataset(imgPathList, boxes_list, pred_cls_list, RESIZE_TO, RESIZE_TO, CLASSES, get_train_transform())

        train_concatenate_dataset = ConcatDataset([train_dataset, inferenceDataset])
        train_concatenate_loader = DataLoader(
            train_concatenate_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            collate_fn=collate_fn
        )

        current_iter += 1
        Engine = engine(train_concatenate_loader, valid_loader, current_iter)
        Engine.train()
        current_best_mAP = Engine.compute_mAP()
        best_mAP_history.append(current_best_mAP)
        print('='*30)
        print(f'current mAP: {current_best_mAP} / last mAP: {last_best_mAP}')
        print(f'mAP history: {best_mAP_history}')
    
    best_Engine.save_loss_plot_and_model()
    best_Engine.compute_IoU()
    best_Engine.compute_mAP()
    best_Engine.print_model_info()
    print(f'mAP history: {best_mAP_history}')
    time_elapsed = time.time() - since
    print(f'Train model complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print('='*30)

if __name__ == '__main__':
    simple_RCNN_training()