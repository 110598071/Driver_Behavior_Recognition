from fast_rcnn.config import RESIZE_TO, DEVICE, NUM_CLASSES, INFERENCE_MODEL, DETECTION_THRESHOLD, CLASSES
from fast_rcnn.model import create_model
from skimage.transform import resize
from skimage.feature import hog
import os
import cv2
import csv
import random
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, random_split
from torchvision import datasets, models, transforms
from torchvision.io import read_image
from sklearn.model_selection import train_test_split

class SimpleImageDataset(Dataset):
    def __init__(self, imgPathAndLabelList, transform=None):
        self.imgPathList = imgPathAndLabelList[0]
        self.labelList = imgPathAndLabelList[1]
        self.transform = transform

    def __len__(self):
        return len(self.imgPathList)

    def __getitem__(self, idx):
        image = Image.open(self.imgPathList[idx])
        label = self.labelList[idx]
        if (self.transform is not None):
            image = self.transform(image)
        return image, label

class ImageDataset(Dataset):
    def __init__(self, imgPathAndLabelList, datasetType, transform=None):
        self.imgPathList = imgPathAndLabelList[0]
        self.labelList = imgPathAndLabelList[1]
        self.transform = transform

        self.train_cls_map = get_cls_map('../csv/AUC/AUC_skeleton_merge_train_data.csv')
        self.test_cls_map = get_cls_map('../csv/AUC/AUC_skeleton_merge_test_data.csv')

        self.model = create_model(num_classes=NUM_CLASSES).to(DEVICE)
        self.model.load_state_dict(torch.load(INFERENCE_MODEL, map_location=DEVICE))
        self.model.eval()

    def __len__(self):
        return len(self.imgPathList)

    def __getitem__(self, idx):
        # im_pil = Image.open(self.imgPathList[idx])
        label = self.labelList[idx]
        im_pil = get_rcnn_mask_image(self.model, self.imgPathList[idx])
        # im_pil = get_sobel_blur_img(self.imgPathList[idx])
        # im_pil = self.get_openpose_mask_img(label, self.imgPathList[idx])

        if (self.transform is not None):
            im_pil = self.transform(im_pil)
        return im_pil, label
    
    def get_openpose_mask_img(self, cls, imgPath):
        img_name = imgPath.split('\\')[-1]
        if 'train' in imgPath:
            keypoint_coordinate = self.train_cls_map[cls][img_name]
        else:
            keypoint_coordinate = self.test_cls_map[cls][img_name]

        x_coordinate = keypoint_coordinate[0:len(keypoint_coordinate):2]
        y_coordinate = keypoint_coordinate[1:len(keypoint_coordinate):2]
        x_min, x_max = np.min(list(filter(lambda coor:coor>=0, x_coordinate))), np.max(x_coordinate)
        y_min, y_max = np.min(list(filter(lambda coor:coor>=0, y_coordinate))), np.max(y_coordinate)

        img = cv2.imread(imgPath)
        (height, width, _) = img.shape
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        x_min = x_min if x_min-30 < 0 else x_min-30
        y_min = y_min if y_min-30 < 0 else y_min-30
        x_max = x_max if x_max+100 > width else x_max+100
        y_max = y_max if y_max+30 > height else y_max+30
        mask = np.zeros(img.shape[:2], np.uint8)
        mask[y_min:y_max, x_min:x_max] = 255

        img = cv2.bitwise_and(img, img, mask = mask)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = np.stack((img,)*3, axis=-1)

        kernel_size = 3
        sigma = 0
        kernel_1d = cv2.getGaussianKernel(kernel_size, sigma)
        for _ in range(20):
            img = cv2.sepFilter2D(img, -1, kernel_1d, kernel_1d)

        x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
        y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
        absX = cv2.convertScaleAbs(x)
        absY = cv2.convertScaleAbs(y)
        img = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

        im_pil = Image.fromarray(img)
        return im_pil

# class ImageDataset(Dataset):
#     def __init__(self, imgPathAndLabelList, datasetType, transform=None):
#         print("="*30)
#         print("preparing dataset")
#         self.imgList = []
#         pbar = tqdm(imgPathAndLabelList[0])
#         for imgPath in pbar:
#             img = Image.open(imgPath)
#             # img = img.crop((85, 100, 575, 383))
#             if (transform is not None):
#                 self.imgList.append(transform(img))
#             else:
#                 self.imgList.append(img)
#             img.close()
#             pbar.set_description(f'{datasetType:5s} progress:')
#         self.labelList = imgPathAndLabelList[1]
#         self.transform = transform
#         print("="*30)
#         print()

#     def __len__(self):
#         return len(self.imgList)

#     def __getitem__(self, idx):
#         image = self.imgList[idx]
#         label = self.labelList[idx]
#         return image, label

def get_dataset_split_by_driver(ORIGIN_IMG_PATH, dataSet):
    imgList, labelList = [], []    
    for dirPath, _, images in os.walk(os.path.join(ORIGIN_IMG_PATH, dataSet, '')):
        for imgName in images:
            imgList.append(os.path.join(dirPath, imgName))
            labelList.append(int(dirPath[-1]))
    return (imgList, labelList)

def get_CIFAR10_dataset():
    downloadPath = 'D:/Project/pytorch-openpose/experiment_0211/dataset'
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    image_datasets = {
        'train': datasets.CIFAR10(root=downloadPath, train=True, download=True, transform=transform),
        'test': datasets.CIFAR10(root=downloadPath, train=False, download=True, transform=transform),
    }

    # train_dataset = datasets.CIFAR10(root=downloadPath, train=True, download=True, transform=transform)
    # dataset_len = train_dataset.__len__()
    # image_datasets = {
    #     'train': torch.utils.data.random_split(train_dataset, [int(dataset_len*0.8), dataset_len-int(dataset_len*0.8)])[0],
    #     'val': torch.utils.data.random_split(train_dataset, [int(dataset_len*0.8), dataset_len-int(dataset_len*0.8)])[1],
    #     'test': datasets.CIFAR10(root=downloadPath, train=False, download=True, transform=transform),
    # }
    return image_datasets

def get_sobel_blur_img(imgPath):
    img = cv2.imread(imgPath)
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

    img = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    return im_pil

def get_canny_blur_img(imgPath):
    img = cv2.imread(imgPath)
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)
    dst = cv2.Canny(gray, 60, 110,apertureSize=3)

    img = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    return im_pil

def get_hog_feature_img(imgPath):
    img = cv2.imread(imgPath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.stack((img,)*3, axis=-1)

    fd, hog_image = hog(img, orientations=8, pixels_per_cell=(2, 2), cells_per_block=(2, 2), visualize=True, channel_axis=-1)
    convert_img = np.zeros_like(img)
    convert_img[:,:,0] = hog_image
    convert_img[:,:,1] = hog_image
    convert_img[:,:,2] = hog_image
    im_pil = Image.fromarray(convert_img)
    return im_pil

def get_rcnn_mask_image(model, imgPath):
    img = cv2.imread(imgPath)
    img = cv2.resize(img, (RESIZE_TO, RESIZE_TO))
    img_copy = img.copy()

    img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB).astype(np.float32)
    img_copy /= 255.0

    img_copy = np.transpose(img_copy, (2, 0, 1)).astype(np.float64)
    img_copy = torch.tensor(img_copy, dtype=torch.float).cuda()
    img_copy = torch.unsqueeze(img_copy, 0)
    with torch.no_grad():
        outputs = model(img_copy)

    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
    if len(outputs[0]['boxes']) != 0:
        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()
        
        boxes = boxes[scores >= DETECTION_THRESHOLD].astype(np.int32)
        pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]
        
        if ('person' in pred_classes[0:len(boxes)]):
            [x_min, y_min, x_max, y_max] = boxes[pred_classes[0:len(boxes)].index('person')]
        else:
            x_min, y_min, x_max, y_max = 0, 0, img.shape[1], img.shape[0]

    mask = np.zeros(img.shape[:2], np.uint8)
    mask[y_min:y_max, x_min:x_max] = 255

    img = cv2.bitwise_and(img, img, mask = mask)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.Canny(img, 60, 110,apertureSize=3)
    img = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel=(3,3),iterations=3)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    im_pil = Image.fromarray(img)
    return im_pil

def get_cls_map(csvPath):
    f = open(csvPath,'r',encoding = 'utf8')
    plots = csv.reader(f, delimiter=',')

    class_map = [{} for _ in range(10)]
    for i, rowList in enumerate(plots):
        if i != 0:
            coordinate_map = class_map[int(rowList[1])]
            float_list = list(np.float_(rowList[2:len(rowList)-2]))
            int_list = [int(i) for i in float_list]
            coordinate_map[rowList[0]] = int_list
    return class_map
