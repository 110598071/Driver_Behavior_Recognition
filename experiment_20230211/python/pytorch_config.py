import train_with_openpose
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from feature_dataset import FeatureDataset
import glob
import pytorch_model
import fast_rcnn.feature_computation
import numpy as np
import Image_dataset
import pytorch_util
from torchvision import datasets
import csv
import random
import os
from tqdm import tqdm
import skeleton_util
from torch.utils.data import ConcatDataset
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE

## ============================================================================================================= ##
CNN_MASK_DIR = '20230508_0537'
CNN_MODEL = '20230703_0619' # ResNet
# CNN_MODEL = '20230604_1641' # EfficientNet
# CNN_MODEL = '20230603_0728' # Swin Transformer Small
## ============================================================================================================= ##
CNN_DATA_DIR = 'D:/Project/pytorch-openpose/experiment_0211/dataset/AUC_processed_merge_rcnn_mask/model_' + CNN_MASK_DIR
# CNN_DATA_DIR = 'D:/Project/pytorch-openpose/experiment_0211/dataset/StateFarm/rcnn_mask'
CNN_MODEL_PATH = 'D:/Project/pytorch-openpose/experiment_0211/model/cnn/model_' + CNN_MODEL + '.pth'
## ============================================================================================================= ##
RCNN_DATA_DIR = 'D:/Project/pytorch-openpose/experiment_0211/dataset/AUC_processed_merge'
# RCNN_DATA_DIR = 'D:/Project/pytorch-openpose/experiment_0211/dataset/StateFarm/origin_img'
## ============================================================================================================= ##
CAMERA2_DATA_DIR = 'D:/Project/pytorch-openpose/experiment_0211/dataset/AUC_processed/Camera 2/train'
## ============================================================================================================= ##
DATA_BALANCE_DIR = 'D:/Project/pytorch-openpose/experiment_0211/dataset/AUC_processed_merge_data_balance'
DATA_OVER_SAMPLING_DIR = 'D:/Project/pytorch-openpose/experiment_0211/dataset/AUC_processed_merge_data_over_sampling'
## ============================================================================================================= ##
STORE_CONCATE_DATASET = False
READ_CONCATE_DATASET_MODEL = 'model_20230704_1224'
CONCATE_TRAIN_DATASET = 'D:/Project/pytorch-openpose/experiment_0211/dataset/CONCATE_DATASET_CSV/CONCATE_TRAIN_DATASET.csv'
CONCATE_TEST_DATASET = 'D:/Project/pytorch-openpose/experiment_0211/dataset/CONCATE_DATASET_CSV/CONCATE_TEST_DATASET.csv'
TEST_IMG_PATH_LIST = 'D:/Project/pytorch-openpose/experiment_0211/dataset/CONCATE_DATASET_CSV/TEST_IMG_PATH_LIST.csv'
## ============================================================================================================= ##
USE_STATEFRAM_TEST_DATASET = False
SPLIT_TRAIN_FOR_VALIDATE_DATASET = False
VALIDATE_DIR = 'D:/Project/pytorch-openpose/experiment_0211/dataset/AUC_validate'
## ============================================================================================================= ##

class ConcatenateFeatures():
    def __init__(self, CNN_DATA_DIR, RCNN_DATA_DIR, CNN_MODEL_PATH):
        self.modelPath = CNN_MODEL_PATH
        self.label_train = []
        self.label_test = []
        self.train_rcnn_imgPathList = []
        self.test_rcnn_imgPathList = []
        self.train_cnn_imgPathList = []
        self.test_cnn_imgPathList = []

        self.keypoints_train_list = get_keypoints_feautres_list('train')
        self.keypoints_test_list = get_keypoints_feautres_list('test')

        for cls in range(10):
            self.label_train.extend([cls for _ in range(len(glob.glob(f"{RCNN_DATA_DIR}/train/c{cls}/*.jpg")))])
            self.label_test.extend([cls for _ in range(len(glob.glob(f"{RCNN_DATA_DIR}/test/c{cls}/*.jpg")))])
            self.train_rcnn_imgPathList.extend(glob.glob(f"{RCNN_DATA_DIR}/train/c{cls}/*.jpg"))
            self.test_rcnn_imgPathList.extend(glob.glob(f"{RCNN_DATA_DIR}/test/c{cls}/*.jpg"))
            self.train_cnn_imgPathList.extend(glob.glob(f"{CNN_DATA_DIR}/train/c{cls}/*.jpg"))
            self.test_cnn_imgPathList.extend(glob.glob(f"{CNN_DATA_DIR}/test/c{cls}/*.jpg"))

        self.data_train = [[] for _ in range(len(self.label_train))]
        self.data_test = [[] for _ in range(len(self.label_test))]

    def concatenate(self, train_feature_list, test_feature_list):
        self.data_train = np.concatenate([self.data_train, train_feature_list], axis=1)
        self.data_test = np.concatenate([self.data_test, test_feature_list], axis=1)

    def add_object_detection_feautre(self): #32
        train_feature_list = fast_rcnn.feature_computation.feature_computation(self.train_rcnn_imgPathList, self.keypoints_train_list, self.label_train)
        test_feature_list = fast_rcnn.feature_computation.feature_computation(self.test_rcnn_imgPathList, self.keypoints_test_list, self.label_test)
        # train_feature_list = [[] for _ in range(len(self.label_train))]
        self.concatenate(train_feature_list, test_feature_list)

    def add_cnn_output_feature(self): #10
        # model = pytorch_model.get_pretrained_ResNet("101")
        model = pytorch_model.get_pretrained_EfficientNet_m()
        # model = pytorch_model.get_pretrained_Swin_Transformer_small()
        train_feature_list = pytorch_util.test_image(model, self.modelPath, self.train_cnn_imgPathList)
        test_feature_list = pytorch_util.test_image(model, self.modelPath, self.test_cnn_imgPathList)
        # train_feature_list = [[] for _ in range(len(self.label_train))]
        self.concatenate(train_feature_list, test_feature_list)
    
    def add_openpose_feature(self): #308
        train_feature_list = []
        test_feature_list = []

        pbar = tqdm(self.train_rcnn_imgPathList)
        for idx, imgPath in enumerate(pbar):
            train_feature_list.append(skeleton_util.compute_feature(self.keypoints_train_list[self.label_train[idx]][imgPath.split('\\')[-1]]))
            pbar.set_description('openpose train features computation')
        pbar = tqdm(self.test_rcnn_imgPathList)
        for idx, imgPath in enumerate(self.test_rcnn_imgPathList):
            test_feature_list.append(skeleton_util.compute_feature(self.keypoints_test_list[self.label_test[idx]][imgPath.split('\\')[-1]]))
            pbar.set_description('openpose test features computation')
        self.concatenate(train_feature_list, test_feature_list)

    def get_concatenate_features(self):
        return np.asarray(self.data_train), np.asarray(self.label_train), np.asarray(self.data_test), np.asarray(self.label_test)
    
    def get_imgPathList(self):
        return self.train_rcnn_imgPathList, self.test_rcnn_imgPathList

def get_keypoints_feautres_list(dataset):
    keypoints_feautres_list = [{} for _ in range(10)]
    f = open('D:/Project/pytorch-openpose/experiment_0211/csv/AUC/AUC_skeleton_merge_'+dataset+'_data_400.csv','r',encoding = 'utf8')
    # f = open('D:/Project/pytorch-openpose/experiment_0211/csv/StateFarm/StateFarm_skeleton_test_data.csv','r',encoding = 'utf8')
    plots = csv.reader(f, delimiter=',')

    for idx, rowList in enumerate(plots):
        if idx != 0:
            keypoints_feautres_list[int(rowList[1])][rowList[0]] = list(np.float_(rowList[2:len(rowList)]))
    f.close()
    return keypoints_feautres_list

def compute_1d_feature():
    concatenate_features = ConcatenateFeatures(CNN_DATA_DIR, RCNN_DATA_DIR, CNN_MODEL_PATH)
    concatenate_features.add_object_detection_feautre()
    concatenate_features.add_cnn_output_feature()
    # concatenate_features.add_openpose_feature()
    return concatenate_features

def get_1d_feature_datasets():
    if STORE_CONCATE_DATASET:
        concatenate_features = compute_1d_feature()
        data_train, label_train, data_test, label_test = concatenate_features.get_concatenate_features()
        _, test_imgPathList = concatenate_features.get_imgPathList()
        store_1d_feature_datasets_to_csv(data_train, label_train, data_test, label_test, test_imgPathList)
    else:
        data_train, label_train, data_test, label_test, test_imgPathList = read_1d_feature_datasets_from_csv()

    if USE_STATEFRAM_TEST_DATASET:
        data_test, label_test, test_imgPathList = read_StateFarm_1d_feature_datasets_from_csv()
    if SPLIT_TRAIN_FOR_VALIDATE_DATASET:
        data_train, label_train, data_test, label_test, test_imgPathList = split_train_for_validate_dataset(data_train, label_train)

    images_datasets = {
        'train': FeatureDataset(data_train, label_train),
        'test': FeatureDataset(data_test, label_test),
    }
    return images_datasets, [test_imgPathList, data_test, label_test]

def get_images_datasets():
    images_datasets = {x: datasets.ImageFolder(os.path.join(CNN_DATA_DIR, x), pytorch_util.get_data_transforms()[x]) for x in ['train', 'test']}
    images_datasets = training_mask_dataset_augmentation(images_datasets)
    # images_datasets = handcraft_data_balance()
    return images_datasets

def store_1d_feature_datasets_to_csv(data_train, label_train, data_test, label_test, test_imgPathList):
    with open(CONCATE_TRAIN_DATASET, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for idx, data in enumerate(data_train):
            row = []
            row.append(str(label_train[idx]))
            row.extend([str(feature) for feature in data])
            writer.writerow(row)
    with open(CONCATE_TEST_DATASET, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for idx, data in enumerate(data_test):
            row = []
            row.append(str(label_test[idx]))
            row.extend([str(feature) for feature in data])
            writer.writerow(row)
    with open(TEST_IMG_PATH_LIST, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(test_imgPathList)

def read_1d_feature_datasets_from_csv():
    data_train = []
    label_train = []
    data_test = []
    label_test = []
    test_imgPathList = []

    with open(CONCATE_TRAIN_DATASET.replace('CONCATE_TRAIN_DATASET', f'{READ_CONCATE_DATASET_MODEL}_CONCATE_TRAIN_DATASET'), newline='') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            label_train.append(int(row[0]))
            feature_list = [float(i) for i in row[1:len(row)]]
            # del feature_list[5]
            # del feature_list[6]
            # del feature_list[6]
            data_train.append(feature_list)
    with open(CONCATE_TEST_DATASET.replace('CONCATE_TEST_DATASET', f'{READ_CONCATE_DATASET_MODEL}_CONCATE_TEST_DATASET'), newline='') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            label_test.append(int(row[0]))
            feature_list = [float(i) for i in row[1:len(row)]]
            # del feature_list[5]
            # del feature_list[6]
            # del feature_list[6]
            data_test.append(feature_list)
    with open(TEST_IMG_PATH_LIST, newline='') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            test_imgPathList.extend(row)
    return np.asarray(data_train), np.asarray(label_train), np.asarray(data_test), np.asarray(label_test), np.asarray(test_imgPathList)

def read_StateFarm_1d_feature_datasets_from_csv():
    STATEFARM_CONCATE_TEST_DATASET = 'D:/Project/pytorch-openpose/experiment_0211/dataset/StateFarm/CONCATE_TEST_DATASET.csv'
    STATEFARM_TEST_IMG_PATH_LIST = 'D:/Project/pytorch-openpose/experiment_0211/dataset/StateFarm/TEST_IMG_PATH_LIST.csv'
    data_test = []
    label_test = []
    test_imgPathList = []

    with open(STATEFARM_CONCATE_TEST_DATASET, newline='') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            label_test.append(int(row[0]))
            feature_list = [float(i) for i in row[1:len(row)]]
            data_test.append(feature_list)
    with open(STATEFARM_TEST_IMG_PATH_LIST, newline='') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            test_imgPathList.extend(row)

    reserve_data_test = []
    reserve_label_test = []
    reserve_test_imgPathList = []
    reserve_idx_list = [i for i in range(0, len(label_test), 8)]
    for idx in range(len(label_test)):
        if idx in reserve_idx_list:
            reserve_data_test.append(data_test[idx])
            reserve_label_test.append(label_test[idx])
            reserve_test_imgPathList.append(test_imgPathList[idx])
    return np.asarray(reserve_data_test), np.asarray(reserve_label_test), np.asarray(reserve_test_imgPathList)

def split_train_for_validate_dataset(data_train, label_train):
    whole_train_data_name_list = []
    for cls in range(10):
        whole_train_data_name_list.append(glob.glob(f"{RCNN_DATA_DIR}/train/c{cls}/*.jpg"))
    for cls in range(10):
        for idx, data_name in enumerate(whole_train_data_name_list[cls]):
            whole_train_data_name_list[cls][idx] = data_name.split('\\')[-1]

    data_test = []
    label_test = []
    test_imgPathList = []
    validate_data_name_list = []
    for cls in range(10):
        label_test.extend([cls for _ in range(len(glob.glob(f"{VALIDATE_DIR}/c{cls}/*.jpg")))])
        test_imgPathList.extend(glob.glob(f"{VALIDATE_DIR}/c{cls}/*.jpg"))
        validate_data_name_list.append(glob.glob(f"{VALIDATE_DIR}/c{cls}/*.jpg"))
    for cls in range(10):
        for idx, data_name in enumerate(validate_data_name_list[cls]):
            validate_data_name_list[cls][idx] = data_name.split('\\')[-1]

    reserve_train_index = []
    accumulation = 0
    for cls in range(10):
        for idx, whole_train_data_name in enumerate(whole_train_data_name_list[cls]):
            if whole_train_data_name not in validate_data_name_list[cls]:
                reserve_train_index.append(idx + accumulation)
        accumulation += len(whole_train_data_name_list[cls])

    reserve_data_train = []
    reserve_label_train = []
    for idx, data in enumerate(data_train):
        if idx in reserve_train_index:
            reserve_data_train.append(data)
            reserve_label_train.append(label_train[idx])
        else:
            data_test.append(data)
    return np.asarray(reserve_data_train), np.asarray(reserve_label_train), np.asarray(data_test), np.asarray(label_test), np.asarray(test_imgPathList)

def training_mask_dataset_augmentation(images_datasets):
    cls_imgPathList = []
    origin_imgPathList = []
    origin_label_train = []
    general_augment_imgPathList = []
    general_augment_label_train = []
    camera2_augment_imgPathList = []
    camera2_augment_label_train = []
    cls_amount = []
    augment_amount = [0 for _ in range(10)]

    for cls in range(10):
        cls_imgPathList.append(glob.glob(f"{CNN_DATA_DIR}/train/c{cls}/*.jpg"))
        cls_amount.append(len(glob.glob(f"{CNN_DATA_DIR}/train/c{cls}/*.jpg")))

    for cls_idx, cls_imgPath in enumerate(cls_imgPathList):
        for imgPath in cls_imgPath:
            file_name = imgPath.split('\\')[-1]
            if os.path.exists(f'{CAMERA2_DATA_DIR}/c{cls_idx}/{file_name}'):
                camera2_augment_imgPathList.extend([imgPath for _ in range(2)])
                camera2_augment_label_train.extend([cls_idx for _ in range(2)])

    max_amount = np.min(cls_amount) * 2
    for idx, amount in enumerate(cls_amount):
        augment_amount[idx] = max_amount - amount

    for cls_idx, amount in enumerate(augment_amount):
        if amount < 0:
            origin_imgPathList.extend(random.sample(cls_imgPathList[cls_idx], max_amount))
            origin_label_train.extend([cls_idx for _ in range(max_amount)])
        else:
            origin_imgPathList.extend(cls_imgPathList[cls_idx])
            origin_label_train.extend([cls_idx for _ in range(cls_amount[cls_idx])])
            general_augment_imgPathList.extend(random.sample(cls_imgPathList[cls_idx], augment_amount[cls_idx]))
            general_augment_label_train.extend([cls_idx for _ in range(augment_amount[cls_idx])])

    # max_amount = np.max(cls_amount)
    # for idx, amount in enumerate(cls_amount):
    #     augment_amount[idx] = max_amount - amount
    # print(augment_amount)
        
    # for cls_idx, amount in enumerate(augment_amount):
    #     origin_imgPathList.extend(cls_imgPathList[cls_idx])
    #     origin_label_train.extend([cls_idx for _ in range(cls_amount[cls_idx])])
    #     if amount != 0:
    #         while amount > 0:
    #             if amount > cls_amount[cls_idx]:
    #                 general_augment_imgPathList.extend(cls_imgPathList[cls_idx])
    #                 general_augment_label_train.extend([cls_idx for _ in range(cls_amount[cls_idx])])
    #             else:
    #                 general_augment_imgPathList.extend(random.sample(cls_imgPathList[cls_idx], amount))
    #                 general_augment_label_train.extend([cls_idx for _ in range(amount)])
    #             amount -= cls_amount[cls_idx]
    # print(len(origin_label_train))
    # print(len(general_augment_label_train))

    origin_dataset = Image_dataset.SimpleImageDataset((origin_imgPathList, origin_label_train), pytorch_util.get_data_transforms()['train'])
    general_augment_dataset = Image_dataset.SimpleImageDataset((general_augment_imgPathList, general_augment_label_train), pytorch_util.get_data_transforms()['augmentation'])
    camera2_augment_dataset = Image_dataset.SimpleImageDataset((camera2_augment_imgPathList, camera2_augment_label_train), pytorch_util.get_data_transforms()['camera2_augmentation'])
    images_datasets['train'] = ConcatDataset([origin_dataset, general_augment_dataset, camera2_augment_dataset])
    return images_datasets

def handcraft_data_balance():
    cls_imgPathList = []
    camera2_augment_imgPathList = []
    camera2_augment_label_train = []

    for cls in range(10):
        cls_imgPathList.append(glob.glob(f"{CNN_DATA_DIR}/train/c{cls}/*.jpg"))

    for cls_idx, cls_imgPath in enumerate(cls_imgPathList):
        for imgPath in cls_imgPath:
            file_name = imgPath.split('\\')[-1]
            if os.path.exists(f'{CAMERA2_DATA_DIR}/c{cls_idx}/{file_name}'):
                camera2_augment_imgPathList.extend([imgPath for _ in range(2)])
                camera2_augment_label_train.extend([cls_idx for _ in range(2)])

    train_balance_imgPathList = []
    train_balance_label = []
    train_over_sampling_imgPathList = []
    train_over_sampling_label = []

    for cls_idx, cls_imgPath in enumerate(cls_imgPathList):
        for imgPath in cls_imgPath:
            file_name = imgPath.split('\\')[-1]
            if os.path.exists(f'{DATA_BALANCE_DIR}/train/c{cls_idx}/{file_name}'):
                train_balance_imgPathList.append(imgPath)
                train_balance_label.append(cls_idx)
            if os.path.exists(f'{DATA_OVER_SAMPLING_DIR}/train/c{cls_idx}/{file_name}'):
                train_over_sampling_imgPathList.append(imgPath)
                train_over_sampling_label.append(cls_idx)

    test_dataset = datasets.ImageFolder(os.path.join(CNN_DATA_DIR, 'test'), pytorch_util.get_data_transforms()['test'])
    camera2_augment_dataset = Image_dataset.SimpleImageDataset((camera2_augment_imgPathList, camera2_augment_label_train), pytorch_util.get_data_transforms()['camera2_augmentation'])
    train_balance_dataset = Image_dataset.SimpleImageDataset((train_balance_imgPathList, train_balance_label), pytorch_util.get_data_transforms()['train'])
    train_over_sampling_dataset = Image_dataset.SimpleImageDataset((train_over_sampling_imgPathList, train_over_sampling_label), pytorch_util.get_data_transforms()['augmentation'])
    images_datasets = {
        'train': ConcatDataset([train_balance_dataset, train_over_sampling_dataset, camera2_augment_dataset]),
        'test': test_dataset,
    }
    return images_datasets