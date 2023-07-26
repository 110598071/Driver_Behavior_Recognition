from fast_rcnn.config import RESIZE_TO, DEVICE, NUM_CLASSES, INFERENCE_MODEL, DETECTION_THRESHOLD, CLASSES, INFERENCE_DIR
from fast_rcnn.feature_computation import APPERENCE_OBJECT, COMPUTE_DISTANCE_OBJECT, COMPUTE_OVERLAP_OBJECT, Point, compute_angle
from fast_rcnn.model import create_model
from fast_rcnn.inference import inference
from fast_rcnn.mAP_computation import mAP_inference, get_ground_true_boxes, compute_mAP
from fast_rcnn.IoU_computation import IoU_computation
from pytorch_config import get_keypoints_feautres_list
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from skimage.transform import resize
from sklearn.metrics import classification_report
from skimage.feature import hog
from sklearn.metrics import confusion_matrix
from datetime import datetime
import seaborn as sns
import pandas as pd
from skimage import exposure
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import torch
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np
import csv
import time
import shutil
import os
import Image_dataset
import random
from collections import Counter
from sklearn.datasets import make_classification
import PIL.Image as Image
import pytorch_model
import AlexNet
import pytorch_util
import pytorch_progress
from tqdm import tqdm
from torchsummary import summary
import glob as glob
import torchvision
import fast_rcnn.feature_computation
import train_with_openpose
import pytorch_config
from feature_dataset import FeatureDataset
from FCLayer import EnsembleModel
from fast_rcnn import engine

def get_data_transforms():
    RESIZE = [400 for _ in range(2)]
    data_transforms = {
        'train': transforms.Compose([
            # transforms.Grayscale(),
            transforms.Resize(RESIZE),
            transforms.ToTensor(),
            # transforms.Lambda(lambda x: x.repeat(3,1,1)),
            transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5)),
            # transforms.ToPILImage(),
        ]),
        'val': transforms.Compose([
            transforms.Resize(RESIZE),
            transforms.ToTensor(),
            # transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5)),
        ]),
        'test': transforms.Compose([
            transforms.Resize(RESIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5)),
        ]),
    }
    return data_transforms

def get_feature_maps():
    img = Image.open("D:/Project/pytorch-openpose/experiment_0211/dataset/AUC_processed_split_merge/train/c0/416.jpg")
    # img = Image.open("D:/Project/pytorch-openpose/experiment_0211/dataset/AUC_processed_split_merge/test/c0/325.jpg")
    new_img = get_data_transforms()['train'](img)
    # new_img.show()

    modelPath = "D:/Project/pytorch-openpose/experiment_0211/model/model_20230320_0702.pth"
    model = pytorch_model.get_original_ResNet("101")
    checkpoint = torch.load(modelPath)
    model.load_state_dict(checkpoint)

    # we will save the conv layer weights in this list
    model_weights =[]
    #we will save the 49 conv layers in this list
    conv_layers = []
    # get all the model children as list
    model_children = list(model.children())
    #counter to keep count of the conv layers
    counter = 0
    #append all the conv layers and their respective wights to the list
    for i in range(len(model_children)):
        if type(model_children[i]) == nn.Conv2d:
            counter+=1
            model_weights.append(model_children[i].weight)
            conv_layers.append(model_children[i])
        elif type(model_children[i]) == nn.Sequential:
            for j in range(len(model_children[i])):
                for child in model_children[i][j].children():
                    if type(child) == nn.Conv2d:
                        counter+=1
                        model_weights.append(child.weight)
                        conv_layers.append(child)
    # print(f"Total convolution layers: {counter}")
    # print("conv_layers")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    image = new_img.unsqueeze(0)
    image = image.to(device)

    outputs = []
    names = []
    for layer in conv_layers[0:]:
        image = layer(image)
        outputs.append(image)
        names.append(str(layer))
    # print(len(outputs))
    # for feature_map in outputs:
    #     print(feature_map.shape)

    processed = []
    for feature_map in outputs:
        feature_map = feature_map.squeeze(0)
        gray_scale = torch.sum(feature_map,0)
        gray_scale = gray_scale / feature_map.shape[0]
        processed.append(gray_scale.data.cpu().numpy())
    # for fm in processed:
    #     print(fm.shape)

    for k in range(0, 100, 20):
        fig = plt.figure(figsize=(30, 50))
        for i in range(20):
            a = fig.add_subplot(5, 4, i+1)
            j = i+k
            imgplot = plt.imshow(processed[j])
            a.axis("off")
            a.set_title(names[i].split('(')[0]+":"+str(j), fontsize=30)
        plt.savefig(str('../plot/feature_map/feature_maps_'+str(int((j+1)/20))+'.jpg'), bbox_inches='tight')

def plot_skimage_hog_image():
    img = cv2.imread("D:/Project/pytorch-openpose/experiment_0211/dataset/AUC_processed_merge/train/c0/1443.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.stack((img,)*3, axis=-1)
    plt.axis("off")
    # plt.imshow(img)
    # plt.show()
    print(img.shape)

    resized_img = resize(img, (224, 224))
    print(resized_img.shape)

    fd, hog_image = hog(resized_img, orientations=8, pixels_per_cell=(2, 2), cells_per_block=(2, 2), visualize=True, channel_axis=-1)
    convert_img = np.zeros_like(resized_img)
    convert_img[:,:,0] = hog_image
    convert_img[:,:,1] = hog_image
    convert_img[:,:,2] = hog_image
    print(convert_img.shape)
    plt.axis("off")
    plt.imshow(convert_img, cmap="gray")
    plt.show()

def plot_cv2_hog_image():
    winSize = (64,64)
    blockSize = (16,16)
    blockStride = (8,8)
    cellSize = (8,8)
    nbins = 8
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    img = cv2.imread("D:/Project/pytorch-openpose/experiment_0211/dataset/AUC_processed_merge/train/c0/1443.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.stack((img,)*3, axis=-1)
    hog_image = hog.compute(img)
    print(hog_image.shape)

    # plt.axis("off")
    # plt.imshow(hog_image)
    # plt.show()

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

    # img = cv2.GaussianBlur(img, (3, 3), 0)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # img = cv2.Canny(img, 60, 110,apertureSize=3)
    # img = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel=(3,3),iterations=3)

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # plt.axis("off")
    # plt.imshow(img)
    # plt.show()

    cv2.imwrite(imgPath.replace('AUC_processed_merge', 'AUC_processed_merge_rcnn_mask/model_20230520_2042'), img)
    # cv2.imwrite(imgPath.replace('origin_img', 'rcnn_mask'), img)

def inference_and_get_rcnn_mask_image():
    model = create_model(num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(INFERENCE_MODEL, map_location=DEVICE))
    model.eval()

    IMG_PATH = "D:/Project/pytorch-openpose/experiment_0211/dataset/AUC_processed_merge"
    # IMG_PATH = "D:/Project/pytorch-openpose/experiment_0211/dataset/StateFarm/origin_img"
    imgPath_list = []
    for dir in ['train', 'test']:
        for cls in range(10):
            imgPath_list.extend(glob.glob(f"{IMG_PATH}/{dir}/c{cls}/*.jpg"))
    # for cls in range(10):
    #     imgPath_list.extend(glob.glob(f"{IMG_PATH}/c{cls}/*.jpg"))
    
    pbar = tqdm(imgPath_list)
    for imgPath in pbar:
        get_rcnn_mask_image(model, imgPath)

def get_roi_align_image(model, imgPath):
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

    pooler = torchvision.ops.RoIAlign(output_size=400, spatial_scale=0.9, sampling_ratio=1)
    img_tensor = torch.tensor(np.asarray([img.reshape((3, 400, 400))])).float()
    box = torch.tensor([[x_min, y_min, x_max, y_max]]).float()
    output = pooler(img_tensor, [box])

    to_pil = transforms.ToPILImage()
    output_pil = to_pil(output[0])
    plt.imshow(output_pil)
    plt.show()
    # output = output.numpy()[0]

    # print(output.shape)
    # print(img.shape)

    # plt.axis("off")
    # plt.imshow(output.reshape((400, 400, 3)))
    # plt.show()

def get_fast_rcnn_feature_computation():
    keypoints_feautres_list = [{} for _ in range(10)]
    f = open('D:/Project/pytorch-openpose/experiment_0211/csv/AUC/AUC_skeleton_merge_test_data_400.csv','r',encoding = 'utf8')
    plots = csv.reader(f, delimiter=',')

    for idx, rowList in enumerate(plots):
        if idx != 0:
            keypoints_feautres_list[int(rowList[1])][rowList[0]] = list(np.float_(rowList[2:len(rowList)]))
    f.close()

    img_name = 84
    label_test = 7
    imgPathList = [f'D:/Project/pytorch-openpose/experiment_0211/dataset/AUC_processed_merge/test/c{label_test}\\{img_name}.jpg']
    imgPathList.append(f'D:/Project/pytorch-openpose/experiment_0211/dataset/AUC_processed_merge/test/c0\\1.jpg')
    imgPathList.append(f'D:/Project/pytorch-openpose/experiment_0211/dataset/AUC_processed_merge/test/c1\\6.jpg')
    imgPathList.append(f'D:/Project/pytorch-openpose/experiment_0211/dataset/AUC_processed_merge/test/c2\\5.jpg')
    final_features = fast_rcnn.feature_computation.feature_computation(imgPathList, keypoints_feautres_list, [label_test, 0, 1, 2])
    print(len(final_features[0]))
    for features in final_features:
        print(features)

def visulaized_rcnn_mask_image():
    ## remember to close imwrite
    model = create_model(num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(INFERENCE_MODEL, map_location=DEVICE))
    model.eval()
    get_rcnn_mask_image(model, 'D:/Project/pytorch-openpose/experiment_0211/dataset/AUC_processed_merge/test/c0/340.jpg')

def copy_semi_supervised_images():
    IMG_PATH = "D:/Project/pytorch-openpose/experiment_0211/dataset/AUC_processed_merge/train"
    imgPathList = []
    for cls in range(10):
        imgPathList.append(glob.glob(f"{IMG_PATH}/c{cls}/*.jpg"))

    for cls_list in imgPathList:
        random_select_list = random.sample(cls_list, 800)
        for select_imgPath in random_select_list:
            shutil.copyfile(select_imgPath, select_imgPath.replace('AUC_processed_merge/train', 'object_detection_test/semi_supervised'))

def get_cnn_wrong_classification_images_and_csv():
    CNN_MODEL = 'model_20230513_0945'
    MASK_IMG_DIR = 'model_20230508_0537'
    
    CREATE_DIR_PATH = 'D:/Project/pytorch-openpose/experiment_0211/dataset/AUC_processed_merge_wrong_classification'
    for cls in range(10):
        os.makedirs(f'{CREATE_DIR_PATH}/{CNN_MODEL}/c{cls}')

    IMG_PATH = f'D:/Project/pytorch-openpose/experiment_0211/dataset/AUC_processed_merge_rcnn_mask/{MASK_IMG_DIR}/test'
    imgPathList = []
    label_list = []
    for cls in range(10):
        cls_list = glob.glob(f"{IMG_PATH}/c{cls}/*.jpg")
        imgPathList.extend(cls_list)
        label_list.extend([cls for _ in range(len(cls_list))])

    model = pytorch_model.get_pretrained_ResNet('101')
    model_path = f'D:/Project/pytorch-openpose/experiment_0211/model/cnn/{CNN_MODEL}.pth'
    output_list = pytorch_util.test_image(model, model_path, imgPathList)

    record_output = []
    record_imgPath = []
    for idx, output in enumerate(output_list):
        if (np.argmax(output) != label_list[idx]):
            copy_path = imgPathList[idx].replace(f'AUC_processed_merge_rcnn_mask/{MASK_IMG_DIR}/test', f'AUC_processed_merge_wrong_classification/{CNN_MODEL}')
            file_name = copy_path.split('\\')[-1]
            shutil.copyfile(imgPathList[idx], copy_path.replace(file_name, str(np.argmax(output))+'_'+file_name))

            record_output.append([str(np.round(op, 2)) for op in output])
            record_imgPath.append(imgPathList[idx].split(f'AUC_processed_merge_rcnn_mask/{MASK_IMG_DIR}/test/')[-1])

    with open(f'{CREATE_DIR_PATH}/{CNN_MODEL}/{CNN_MODEL}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        first_row = ['cls', 'image', 'pred']
        for i in range(10):
            first_row.append(f'c{i}')
        writer.writerow(first_row)
        for idx, imgPath in enumerate(record_imgPath):
            row = []
            row.append(f'{imgPath[1]}')
            row.append(imgPath.split('.jpg')[0].split('\\')[-1])
            row.append(str(np.argmax([float(op) for op in record_output[idx]])))
            row.extend(record_output[idx])
            writer.writerow(row)

def visialized_rcnn_canny_image():
    img = cv2.imread('D:/Project/pytorch-openpose/experiment_0211/dataset/AUC_processed_merge_rcnn_mask/test/c0/340.jpg')
    print(img.shape)
    img2 = np.zeros_like(img)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.Canny(img, 60, 110,apertureSize=3)
    img = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel=(3,3),iterations=3)
    
    img2[:,:,0] = img
    img2[:,:,1] = img
    img2[:,:,2] = img
    print(img2.shape)
    plt.axis("off")
    plt.imshow(img2)
    plt.show()

def get_object_features_computation_csv():
    DATA_DIR = 'D:/Project/pytorch-openpose/experiment_0211/dataset/AUC_processed_merge'
    label_train = []
    label_test = []
    train_imgPathList = []
    test_imgPathList = []

    keypoints_train_list = get_keypoints_feautres_list('train')
    keypoints_test_list = get_keypoints_feautres_list('test')

    for cls in range(10):
        label_train.extend([cls for _ in range(len(glob.glob(f"{DATA_DIR}/train/c{cls}/*.jpg")))])
        label_test.extend([cls for _ in range(len(glob.glob(f"{DATA_DIR}/test/c{cls}/*.jpg")))])
        train_imgPathList.extend(glob.glob(f"{DATA_DIR}/train/c{cls}/*.jpg"))
        test_imgPathList.extend(glob.glob(f"{DATA_DIR}/test/c{cls}/*.jpg"))

    train_feature_list = fast_rcnn.feature_computation.feature_computation(train_imgPathList, keypoints_train_list, label_train)
    test_feature_list = fast_rcnn.feature_computation.feature_computation(test_imgPathList, keypoints_test_list, label_test)

    first_row = ['cls', 'img']
    first_row.extend(APPERENCE_OBJECT)
    for feature in COMPUTE_DISTANCE_OBJECT:
        merge = feature[0] + '/' + feature[1]
        first_row.append(merge)
    for feature in COMPUTE_OVERLAP_OBJECT:
        merge = feature[0] + '/' + feature[1]
        first_row.append(merge)

    with open('D:/Project/pytorch-openpose/experiment_0211/dataset/train_object_feature.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(first_row)
        for idx, feature in enumerate(train_feature_list):
            row = []
            row.append(label_train[idx])
            row.append(train_imgPathList[idx].split('\\')[-1].split('.jpg')[0])
            row.extend(feature)
            writer.writerow(row)

    with open('D:/Project/pytorch-openpose/experiment_0211/dataset/test_object_feature.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(first_row)
        for idx, feature in enumerate(test_feature_list):
            row = []
            row.append(label_test[idx])
            row.append(test_imgPathList[idx].split('\\')[-1].split('.jpg')[0])
            row.extend(feature)
            writer.writerow(row)

def find_voting_best_weight():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    simpleFC = pytorch_model.get_SimpleFC(38)
    checkpoint = torch.load("D:/Project/pytorch-openpose/experiment_0211/model/cnn/model_20230428_1534.pth")
    simpleFC.load_state_dict(checkpoint)

    data_test = []
    label_test = []
    cnn_output = []
    with open('D:/Project/pytorch-openpose/experiment_0211/dataset/combine_dataset.csv', newline='') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            label_test.append(int(row[0]))
            data_test.append([float(i) for i in row[2:len(row)]])
            cnn_output.append([float(i) for i in row[30:len(row)]])

    toTensor = transforms.ToTensor()
    output_list = []
    pbar = tqdm(data_test)
    for data in pbar:
        data = toTensor(np.asarray([data])).unsqueeze(0).to(torch.float32).to(device)
        output = simpleFC(data)
        output_cpu = output.cpu().detach().numpy().tolist()[0][0][0]
        output_list.append(output_cpu)
        pbar.set_description('get wrong cliassification img')

    best_weight = 0.0
    best_final_output = []
    best_accuracy = 0.0
    pbar = tqdm(np.arange(0.0, 1.0, 0.001))
    for weight in pbar:
        weight = np.round(weight, 3)
        final_output = []
        for idx, output in enumerate(output_list):
            merge_output = []
            for output_idx in range(10):
                merge_output.append(output[output_idx]*weight + cnn_output[idx][output_idx] * (1-weight))
            final_output.append(np.argmax(merge_output))

        correct = 0
        for idx, output in enumerate(final_output):
            if (output == label_test[idx]):
                correct += 1
        acc = np.round(correct / len(label_test), 4)
        if (acc > best_accuracy):
            best_accuracy = acc
            best_weight = weight
            best_final_output = final_output
        pbar.set_description('finding weight')

    print('='*30)
    print(f'best weight: {best_weight}')
    print(f'best accuracy: {best_accuracy}')
    print('='*30)

    cf_matrix = confusion_matrix(label_test, best_final_output)
    ticks = [list(range(10))]
    df = pd.DataFrame(cf_matrix, columns=ticks, index=ticks)
    current_time = datetime.now().strftime("_%Y%m%d_%H%M")

    plt.figure()
    sns.heatmap(df, fmt='', annot=True)
    plt.title('confusion_matrix' + current_time)
    plt.ylabel('actual label')
    plt.xlabel('predicted label')
    plt.savefig('../plot/Confusion/feature_voting/Confusion' + current_time + ".jpg", dpi=300)

def compute_RCNN_IoU():
    model = create_model(num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(INFERENCE_MODEL, map_location=DEVICE))
    model.eval()

    iter_IoU_dict, best_detection_threshold = IoU_computation(model)
    print('='*30)
    print(f'best_detection_threshold: {best_detection_threshold}')
    print()
    for key, value in iter_IoU_dict.items():
        print(f'{key}: {value}')

def rcnn_inference_whole_test_images():
    model = create_model(num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(INFERENCE_MODEL, map_location=DEVICE))
    model.eval()

    test_images = []
    if len(glob.glob(f'{INFERENCE_DIR}/*.jpg')) == 0:
        for subdir in os.listdir(INFERENCE_DIR):
            test_images.extend(glob.glob(f"{INFERENCE_DIR}/{subdir}/*.jpg"))
    else:
        test_images.extend(glob.glob(f"{INFERENCE_DIR}/*.jpg"))
    print(f"Test instances: {len(test_images)}")
    inference(model, test_images, True)

def train_fast_rcnn():
    # PYTORCH_SEED = random.randint(0, 4294967295)
    PYTORCH_SEED = 4290731149
    pytorch_util.setup_seed(PYTORCH_SEED)

    engine.semi_Supervised_Learning()
    # engine.simple_RCNN_training()
    print(PYTORCH_SEED)

def reserve_specific_amount_cheat_fast_rcnn_data():
    CHEAT_DIR = 'D:/Project/pytorch-openpose/experiment_0211/dataset/object_detection_test/cheat'
    image_paths = []
    for subdir in os.listdir(CHEAT_DIR):
        image_paths.append(glob.glob(f"{CHEAT_DIR}/{subdir}/*.jpg"))
    for cls_img_path in image_paths:
        remove_list = random.sample(cls_img_path, 15)
        for remove_img in remove_list:
            os.remove(remove_img)
            os.remove(remove_img.replace('.jpg', '.xml'))

def plot_learning_rate_scheduler():
    TRANSFER_FC_EPOCHS = 50
    TRANSFER_FC_LR = 0.00365

    # TRANSFER_WHOLE_EPOCHS = 30
    # TRANSFER_WHOLE_LR = 0.000001

    transfer_fc_scheduler_kwargs = {
        'max_lr': TRANSFER_FC_LR,
        'pct_start': 0.72,
        'div_factor': 39,
        'final_div_factor': 20.8,
        'total_steps': TRANSFER_FC_EPOCHS,
        'verbose': False,
    }

    # transfer_whole_scheduler_kwargs = {
    #     'max_lr': TRANSFER_WHOLE_LR,
    #     'pct_start': 0.3,
    #     'div_factor': 4,
    #     'final_div_factor': 2,
    #     'total_steps': TRANSFER_WHOLE_EPOCHS,
    #     'verbose': False,
    # }

    simpleFC = pytorch_model.get_MLP()
    # optimizer = torch.optim.SGD(simpleFC.parameters(), lr=0.000505, momentum=0.9, weight_decay=0.005)
    # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    fc_optimizer = torch.optim.Adam(simpleFC.parameters(), lr=TRANSFER_FC_LR)
    fc_lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(fc_optimizer, **transfer_fc_scheduler_kwargs)

    # whole_optimizer = torch.optim.Adam(simpleFC.parameters(), lr=TRANSFER_WHOLE_LR)
    # whole_lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(whole_optimizer, **transfer_whole_scheduler_kwargs)

    idx = [i+1 for i in range(50)]
    lr_list = []
    for i in range(50):
        print(f'lr: {fc_lr_scheduler.get_last_lr()[0]:.4e}')
        lr_list.append(fc_lr_scheduler.get_last_lr()[0])
        fc_lr_scheduler.step()

    plt.figure(figsize=(12, 8), dpi=100, linewidth=2)
    plt.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')
    plt.plot(idx, lr_list, 'o-')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title('ExponentialLR', fontsize=25)
    plt.xlabel("epochs", fontsize=20, color='g')
    plt.ylabel('learning rate', fontsize=20, color='g')
    plt.show()

def get_statefarm_concate_test_dataset():
    concatenate_features = pytorch_config.compute_1d_feature()
    data_train, label_train, data_test, label_test = concatenate_features.get_concatenate_features()
    _, test_imgPathList = concatenate_features.get_imgPathList()

    CONCATE_TEST_DATASET = 'D:/Project/pytorch-openpose/experiment_0211/dataset/StateFarm/CONCATE_TEST_DATASET.csv'
    TEST_IMG_PATH_LIST = 'D:/Project/pytorch-openpose/experiment_0211/dataset/StateFarm/TEST_IMG_PATH_LIST.csv'

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

def read_statefarm_concate_test_dataset(driver_list):
    CONCATE_TEST_DATASET = 'D:/Project/pytorch-openpose/experiment_0211/dataset/StateFarm/CONCATE_TEST_DATASET.csv'
    TEST_IMG_PATH_LIST = 'D:/Project/pytorch-openpose/experiment_0211/dataset/StateFarm/TEST_IMG_PATH_LIST.csv'

    data_train = []
    label_train = []
    data_test = []
    label_test = []
    test_imgPathList = []

    with open(CONCATE_TEST_DATASET, newline='') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            label_test.append(int(row[0]))
            feature_list = [float(i) for i in row[1:len(row)]]
            data_test.append(feature_list)
    with open(TEST_IMG_PATH_LIST, newline='') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            test_imgPathList.extend(row)
    data_test, label_test = split_test_data(data_test, label_test, test_imgPathList, driver_list)
    return np.asarray(data_train), np.asarray(label_train), np.asarray(data_test), np.asarray(label_test), np.asarray(test_imgPathList)

def split_test_data(data_test, label_test, test_imgPathList, driver_list):
    CSV_PATH = 'D:/Project/pytorch-openpose/experiment_0211/dataset/StateFarm/driver_imgs_list.csv'
    img_driver_dict = {}
    driver_amount_dict = {}
    with open(CSV_PATH, newline='') as csvfile:
        rows = csv.reader(csvfile)
        for idx, row in enumerate(rows):
            if idx != 0:
                img_driver_dict[row[2]] = row[0]
                if row[0] not in driver_amount_dict:
                    driver_amount_dict[row[0]] = 0
                driver_amount_dict[row[0]] += 1

    reserve_idx_list = []
    for idx, test_imgPath in enumerate(test_imgPathList):
        file_name = test_imgPath.split('\\')[-1]
        if img_driver_dict[file_name] in driver_list:
            reserve_idx_list.append(idx)
    split_data_test = []
    split_label_test = []
    for reserve_idx in reserve_idx_list:
        split_data_test.append(data_test[reserve_idx])
        split_label_test.append(label_test[reserve_idx])
    return split_data_test, split_label_test

def test_generalization_of_fc_model_for_statefarm():
    driver_list = [['p002', 'p012', 'p014', 'p015', 'p016',
                    'p021', 'p022', 'p024', 'p026', 'p035',
                    'p039', 'p041', 'p042', 'p045', 'p047',
                    'p049', 'p050', 'p051', 'p052', 'p056',
                    'p061', 'p064', 'p066', 'p072', 'p075',
                    'p081']]
    # driver_list = [['p041', 'p012', 'p014', 'p015', 'p016', 'p021', 'p022']]
    # driver_list = [['p061', 'p081']]
    for driver in driver_list:
        data_train, label_train, data_test, label_test, test_imgPathList = read_statefarm_concate_test_dataset(driver)

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model = pytorch_model.get_SimpleFC()
        modelPath = "D:/Project/pytorch-openpose/experiment_0211/model/cnn/model_20230519_1909.pth"
        checkpoint = torch.load(modelPath)
        model.load_state_dict(checkpoint)
        model.eval()

        toTensor = transforms.ToTensor()
        output_list = []
        pbar = tqdm(data_test)
        for data in pbar:
            data = toTensor(np.asarray([data])).squeeze(0).to(torch.float32).to(device)
            output = model(data)
            output_cpu = output.cpu().detach().numpy().tolist()[0]
            output_list.append(np.argmax(output_cpu))
            pbar.set_description('get wrong cliassification img')
        
        cnt = 0
        for idx, pred in enumerate(output_list):
            if pred == label_test[idx]:
                cnt += 1
        print(f'accuracy: {(cnt/len(label_test)):.4f}')

        cf_matrix = confusion_matrix(label_test, output_list)
        rowSum = cf_matrix.sum(axis=1)
        cf_matrix = np.array([np.round(row / rowSum[i] * 100, 1) for i, row in enumerate(cf_matrix)])
        ticks = [list(range(10))]
        df = pd.DataFrame(cf_matrix, columns=ticks, index=ticks)

        plt.figure()
        sns.heatmap(df, fmt='', annot=True)
        plt.title('confusion_matrix')
        plt.ylabel('actual label')
        plt.xlabel('predicted label')
        plt.show()

def test_fc_model():
    since = time.time()
    concatenate_features = pytorch_config.compute_1d_feature()
    data_train, label_train, data_test, label_test = concatenate_features.get_concatenate_features()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = pytorch_model.get_SLP()
    modelPath = "D:/Project/pytorch-openpose/experiment_0211/model/cnn/model_20230519_1909.pth"
    checkpoint = torch.load(modelPath)
    model.load_state_dict(checkpoint)
    model.eval()

    toTensor = transforms.ToTensor()
    output_list = []
    pbar = tqdm(data_test)
    for data in pbar:
        data = toTensor(np.asarray([data])).squeeze(0).to(torch.float32).to(device)
        output = model(data)
        output_cpu = output.cpu().detach().numpy().tolist()[0]
        output_list.append(np.argmax(output_cpu))
        pbar.set_description('FC Inference')
    
    cnt = 0
    for idx, pred in enumerate(output_list):
        if pred == label_test[idx]:
            cnt += 1
    print(f'accuracy: {(cnt/len(label_test)):.4f}')
    time_elapsed = time.time() - since
    print(f'Test model complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    cf_matrix = confusion_matrix(label_test, output_list)
    rowSum = cf_matrix.sum(axis=1)
    cf_matrix = np.array([np.round(row / rowSum[i] * 100, 1) for i, row in enumerate(cf_matrix)])
    ticks = [list(range(10))]
    df = pd.DataFrame(cf_matrix, columns=ticks, index=ticks)

    plt.figure()
    sns.heatmap(df, fmt='', annot=True)
    plt.title('confusion_matrix')
    plt.ylabel('actual label')
    plt.xlabel('predicted label')
    plt.show()

def get_fc_classification_report():
    data_train, label_train, data_test, label_test, test_imgPathList = pytorch_config.read_1d_feature_datasets_from_csv()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = pytorch_model.get_MLP()
    modelPath = "D:/Project/pytorch-openpose/experiment_0211/model/cnn/model_20230608_1739.pth"
    checkpoint = torch.load(modelPath)
    model.load_state_dict(checkpoint)
    model.eval()

    toTensor = transforms.ToTensor()
    output_list = []
    pbar = tqdm(data_test)
    for data in pbar:
        data = toTensor(np.asarray([data])).squeeze(0).to(torch.float32).to(device)
        output = model(data)
        output_cpu = output.cpu().detach().numpy().tolist()[0]
        output_list.append(np.argmax(output_cpu))
        pbar.set_description('FC Inference')

    print('=' * 30)
    target_names = ['c'+str(idx) for idx in range(10)]
    print(classification_report(label_test, output_list, target_names=target_names, digits=4))
    print('=' * 30)

def random_forest_for_feature_classification_and_adjusted_weight():
    data_train, label_train, data_test, label_test, test_imgPathList = pytorch_config.read_1d_feature_datasets_from_csv()
    feature_data_train = []
    feature_data_test = []
    resnet_data_train = []
    resnet_data_test = []

    for data in data_train:
        feature_data_train.append(data[0:len(data)-10])
        resnet_data_train.append(data[len(data)-10:len(data)])
    for data in data_test:
        feature_data_test.append(data[0:len(data)-10])
        resnet_data_test.append(data[len(data)-10:len(data)])

    feature_data_train = np.asarray(feature_data_train)
    feature_data_test = np.asarray(feature_data_test)
    resnet_data_train = np.asarray(resnet_data_train)
    resnet_data_test = np.asarray(resnet_data_test)

    # seed = random.randint(0, 4294967295)
    seed = 2103327929
    clf = RandomForestClassifier(n_estimators=150, random_state=seed)
    clf.fit(feature_data_train, label_train)
    score = clf.score(feature_data_test, label_test)
    print(seed)
    print(score)
    print("="*30)

    pred_list = clf.predict_proba(feature_data_test)
    best_weight = 0.0
    best_accuracy = 0.0
    pbar = tqdm(np.arange(0.0, 1.0, 0.001))
    for weight in pbar:
        weight = np.round(weight, 3)
        final_list = []
        for idx, pred in enumerate(pred_list):
            final_list.append(np.argmax([proba*weight + resnet_data_test[idx][proba_idx]*(1-weight) for proba_idx, proba in enumerate(pred)]))
        
        cnt = 0
        for idx, final_pred in enumerate(final_list):
            if final_pred == label_test[idx]:
                cnt += 1
        accuracy = np.round((cnt/len(label_test)), 4)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_weight = weight
        pbar.set_description('finding weight:')
    
    print(f'best weight: {best_weight:.3f}')
    print(f'best accuracy: {best_accuracy:.4f}')

def mAP_computation():
    model = create_model(num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(INFERENCE_MODEL, map_location=DEVICE))
    model.eval()

    test_images = []
    INFERENCE_DIR = 'D:/Project/pytorch-openpose/experiment_0211/dataset/object_detection_test/test'
    for subdir in os.listdir(INFERENCE_DIR):
        test_images.extend(glob.glob(f"{INFERENCE_DIR}/{subdir}/*.jpg"))
    print(f"Test instances: {len(test_images)}")

    # test_images = []
    # INFERENCE_DIR = 'D:/Project/pytorch-openpose/experiment_0211/dataset/object_detection_test/test'
    # test_images.extend(glob.glob(f"{INFERENCE_DIR}/c1/*.jpg"))

    # test_images = ['D:/Project/pytorch-openpose/experiment_0211/dataset/object_detection_test/test/c0/010.jpg',
    #                'D:/Project/pytorch-openpose/experiment_0211/dataset/object_detection_test/test/c0/011.jpg']
    # test_images = ['D:/Project/pytorch-openpose/experiment_0211/dataset/object_detection_test/test/c0/011.jpg']

    pred_boxes_list, pred_cls_list, pred_scores_list = mAP_inference(model, test_images)
    ground_true_boxes_list, ground_true_cls_list = get_ground_true_boxes(test_images)
    mAP, precision_list, recall_list, cls_list, AP_list = compute_mAP(pred_boxes_list, pred_cls_list, pred_scores_list, ground_true_boxes_list, ground_true_cls_list)

    print(f'mAP: {mAP}')
    print(f'avg Precision: {np.round(np.mean(precision_list), 4)}')
    print(f'avg Recall: {np.round(np.mean(recall_list), 4)}')
    for idx, cls in enumerate(cls_list):
        print(f'{("【"+cls+"】"):18s} AP: {AP_list[idx]:.4f} / precision: {precision_list[idx]:.3f} / recall: {recall_list[idx]:.3f}')

if __name__ == '__main__':
    # get_fast_rcnn_feature_computation()
    # get_cnn_wrong_classification_images_and_csv()
    # train_fast_rcnn()
    # inference_and_get_rcnn_mask_image()
    # rcnn_inference_whole_test_images()
    # reserve_specific_amount_cheat_fast_rcnn_data()
    # get_statefarm_concate_test_dataset()
    # test_generalization_of_fc_model_for_statefarm()
    # test_fc_model()
    # mAP_computation()
    # compute_RCNN_IoU()
    # get_fc_classification_report()
    # plot_learning_rate_scheduler()

    kk = [13, 4, 6, 3, 5, 4, 3, 3, 3, 6, 3, 6, 6, 5, 5, 6]
    # kk = [7, 15, 36, 39, 40, 41]
    print(np.sort(kk))
    print(np.quantile(kk, .25))
    print(np.quantile(kk, .5))
    print(np.quantile(kk, .75))

    print(np.quantile(kk, .75) + 1.5*(np.quantile(kk, .75) - np.quantile(kk, .25)))
    print(np.quantile(kk, .25) - 1.5*(np.quantile(kk, .75) - np.quantile(kk, .25)))
    sns.boxplot(kk)
    plt.show()