import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import time
import copy
import os
from PIL import Image
import numpy as np
from datetime import datetime
import Image_dataset
import pytorch_util
from tqdm import tqdm
import seaborn as sns
import pandas as pd
import csv
from sklearn.metrics import confusion_matrix
from feature_dataset import FeatureDataset
import train_with_openpose
import pytorch_config
import shutil
import random
import glob
from pytorch_config import CONCATE_TRAIN_DATASET, CONCATE_TEST_DATASET, STORE_CONCATE_DATASET

class pytorch_model(object):
    def __init__(self, model, BATCH_SIZE, needEarlyStopping=False):
        self.time = ''
        self.since = 0
        self.best_epochs = 1
        self.epochs_history = 0
        self.test_accuracy = 0.0

        self.model = model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.CrossEntropyLoss()
        self.earlyStopping = pytorch_util.EarlyStopping()
        self.needEarlyStopping = needEarlyStopping

        self.trainLoss = []
        self.trainAccuracy = []
        self.valLoss = []
        self.valAccuracy = []
        self.y_pred = []
        self.y_true = []

        train_dataloaders_kwargs = {'batch_size': BATCH_SIZE, 'shuffle': True, 'num_workers': 4, 'pin_memory': True, 'worker_init_fn': pytorch_util.worker_init_fn}
        test_dataloaders_kwargs = {'batch_size': BATCH_SIZE, 'shuffle': False, 'num_workers': 4, 'pin_memory': True, 'worker_init_fn': pytorch_util.worker_init_fn}

        # images_datasets, test_imgPath_data_label_list = pytorch_config.get_1d_feature_datasets()
        # self.test_imgPath_data_label_list = test_imgPath_data_label_list
        images_datasets = pytorch_config.get_images_datasets()

        self.dataloaders = {x: torch.utils.data.DataLoader(images_datasets[x], **(train_dataloaders_kwargs if x=='train' else test_dataloaders_kwargs)) for x in ['train', 'test']}
        self.dataset_sizes = {x: len(images_datasets[x]) for x in ['train', 'test']}
        self.since = time.time()

    def train(self, optimizer, scheduler, needScheduler, num_epochs):
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0
        best_loss = 0.0

        for epoch in range(num_epochs):
            if needScheduler:
                print(f'Epoch {epoch + 1}/{num_epochs}  #learning rate: {scheduler.get_last_lr()[0]:.2e}')
            else:
                print(f'Epoch {epoch + 1}/{num_epochs}')
            print('=' * 30)

            epoch_loss = 0.0
            epoch_acc = 0.0
            for phase in ['train', 'test']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                running_loss = 0.0
                running_corrects = 0
                running_count = 0

                pbar = tqdm(self.dataloaders[phase])
                for inputs, labels in pbar:
                    labels = labels.type(torch.LongTensor)
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        # if phase == 'train':
                        #     outputs, _ = self.model(inputs)
                        # else:
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_count += inputs.size(0)
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += (preds == labels.data).sum().item()
                    pbar.set_description(
                        f'{phase:5s} Loss: {(running_loss / running_count):.4f}  Acc: {(running_corrects / running_count):.4f}')

                if needScheduler and phase == 'train' and epoch != num_epochs - 1:
                    scheduler.step()

                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = float(running_corrects) / self.dataset_sizes[phase]

                if phase == 'train':
                    self.trainAccuracy.append(float(f'{epoch_acc:.4f}'))
                    self.trainLoss.append(float(f'{epoch_loss:.4f}'))
                else:
                    self.valAccuracy.append(float(f'{epoch_acc:.4f}'))
                    self.valLoss.append(float(f'{epoch_loss:.4f}'))

                if phase == 'test' and epoch_acc > best_acc:
                    self.best_epochs = self.epochs_history + epoch+1
                    best_acc = epoch_acc
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(self.model.state_dict())
            torch.cuda.empty_cache()
            print('=' * 30)
            print()
            if self.needEarlyStopping and self.earlyStopping.check(epoch_loss) == 3:
                break

        print("Train and Test Loss/Acc")
        print('=' * 30)
        print(f'Train Loss: {self.trainLoss[-1]}  Accuracy: {self.trainAccuracy[-1]}')
        print(f'Val   Loss: {best_loss:.4f}  Accuracy: {best_acc:.4f}')
        print('=' * 30)
        print()
        self.epochs_history += (epoch+1)
        self.model.load_state_dict(best_model_wts)

    def test(self):
        print('=' * 30)
        running_corrects = 0
        running_count = 0
        pbar = tqdm(self.dataloaders['test'])
        for inputs, labels in pbar:
            labels = labels.type(torch.LongTensor)
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            with torch.no_grad():
                outputs = self.model(inputs)
                pred = torch.argmax(outputs, 1)

            self.y_pred.extend(pred.view(-1).detach().cpu().numpy())
            self.y_true.extend(labels.view(-1).detach().cpu().numpy())

            running_count += inputs.size(0)
            running_corrects += (pred == labels).sum().float()
            pbar.set_description(f'test Accuracy: {(running_corrects / running_count):.4f}')
        self.test_accuracy = np.round((running_corrects / running_count).cpu().numpy(), 4)
        print('=' * 30)
        print(f'Test Accuracy: {(running_corrects / running_count):.4f}  occur epoch: {self.best_epochs}/{self.epochs_history}')
        print('=' * 30)
        print()

    def print_model_use_time(self):
        self.time = datetime.now().strftime("_%Y%m%d_%H%M")
        time_elapsed = time.time() - self.since
        print(f'Train model complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print('=' * 30)

    def save_model(self):
        MODEL_FILEPATH = '../model/cnn/model' + self.time + '.pth'
        torch.save(self.model.state_dict(), MODEL_FILEPATH)
        print("model" + self.time)
        print('=' * 30)

    def plot_training_progress(self):
        plt.style.use('ggplot')
        for plotType in ['Accuracy', "Loss"]:
            start_idx = 1
            if plotType == 'Accuracy':
                accLoss = (self.trainAccuracy, self.valAccuracy)
            else:
                handle_valLoss = list(filter(lambda loss:loss<=3, self.valLoss))
                split_idx = self.valLoss.index(handle_valLoss[0])
                start_idx += split_idx
                handle_trainLoss = self.trainLoss[split_idx:len(self.trainLoss)]
                accLoss = (handle_trainLoss, handle_valLoss)

            train_xtick = [i + start_idx for i in range(len(accLoss[0]))]
            validate_xtick = [i + start_idx for i in range(len(accLoss[1]))]
            plt.figure(figsize=(12, 8), dpi=100, linewidth=2)
            plt.plot(train_xtick, accLoss[0], 's-', color='r', label="train")
            plt.plot(validate_xtick, accLoss[1], 'o-', color='g', label="validate")

            plt.title(plotType)
            plt.xlabel("epochs")
            plt.ylabel(plotType)

            plt.legend(loc="best")
            plt.savefig('../plot/' + plotType + '/' + plotType + self.time + ".jpg")

    def plot_confusion_matrix(self):
        cf_matrix = confusion_matrix(self.y_true, self.y_pred)

        for matrixType in ['amount', 'percentage']:
            if matrixType == 'percentage':
                rowSum = cf_matrix.sum(axis=1)
                cf_matrix = np.array([np.round(row / rowSum[i] * 100, 1) for i, row in enumerate(cf_matrix)])

            ticks = [list(range(10))]
            df = pd.DataFrame(cf_matrix, columns=ticks, index=ticks)

            plt.figure()
            sns.heatmap(df, fmt='', annot=True)
            plt.title('confusion_matrix' + self.time)
            plt.ylabel('actual label')
            plt.xlabel('predicted label')
            plt.savefig('../plot/Confusion/Confusion' + self.time + "_" + matrixType + ".jpg", dpi=300)

    def get_model(self):
        return self.model
    
    def create_wrong_classification_folder(self):
        IMG_PATH = "D:/Project/pytorch-openpose/experiment_0211/dataset/AUC_processed_merge_wrong_classification"
        for cls in range(10):
            os.makedirs(f'{IMG_PATH}/model{self.time}/c{cls}')

    def rename_concate_dataset_csv(self):
        os.rename(CONCATE_TRAIN_DATASET, CONCATE_TRAIN_DATASET.replace('CONCATE_TRAIN_DATASET', f'model{self.time}_CONCATE_TRAIN_DATASET'))
        os.rename(CONCATE_TEST_DATASET, CONCATE_TEST_DATASET.replace('CONCATE_TEST_DATASET', f'model{self.time}_CONCATE_TEST_DATASET'))
    
    def get_wrong_classification_img(self):
        if STORE_CONCATE_DATASET:
            self.rename_concate_dataset_csv()
        self.create_wrong_classification_folder()

        self.model.eval()
        test_imgPathList, data_test, label_test = self.test_imgPath_data_label_list

        toTensor = transforms.ToTensor()
        output_list = []
        record_file_name = []
        record_ground_true = []
        record_output = []
        pbar = tqdm(data_test)
        for data in pbar:
            data = toTensor(np.asarray([data])).squeeze(0).to(torch.float32).to(self.device)
            output = self.model(data)
            output_cpu = output.cpu().detach().numpy().tolist()[0]
            output_list.append(output_cpu)
            pbar.set_description('get wrong cliassification img')
        for idx, output in enumerate(output_list):
            output_idx = np.argmax(output)
            if (output_idx != label_test[idx]):
                copy_path = test_imgPathList[idx].replace('AUC_processed_merge/test', 'AUC_processed_merge_wrong_classification/model'+self.time)
                file_name = copy_path.split('\\')[-1]
                shutil.copyfile(test_imgPathList[idx], copy_path.replace(file_name, str(output_idx)+'_'+file_name))
                record_output.append([str(np.round(op, 2)) for op in output])
                record_file_name.append(file_name)
                record_ground_true.append(label_test[idx])

        with open(f'D:/Project/pytorch-openpose/experiment_0211/dataset/AUC_processed_merge_wrong_classification/model{self.time}/model{self.time}.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            first_row = ['cls', 'image', 'pred']
            for i in range(10):
                first_row.append(f'c{i}')
            writer.writerow(first_row)
            for idx, r_output in enumerate(record_output):
                row = []
                row.append(f'{record_ground_true[idx]}')
                row.append(f'{record_file_name[idx].replace(".jpg", "")}')
                row.append(str(np.argmax([float(op) for op in r_output])))
                row.extend(r_output)
                writer.writerow(row)
        print('=' * 30)

    def get_test_accuracy(self):
        return self.test_accuracy