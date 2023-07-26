from sklearn.ensemble import RandomForestClassifier
import cv2
import matplotlib.pyplot as plt
import numpy as np
import csv
import shutil
import time
import skeleton_util

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
sys.path.append("../..")
from src import model
from src import util
from src.body import Body
from src.hand import Hand

def get_skeleton_data_by_img(dataSet):
    action_data = []
    action_label = []
    imgTotal = 0
    imgCount = 0
    
    MODEL_PATH = "../../model/body_pose_model.pth"
    ORIGIN_IMG_PATH = "D:/Project/pytorch-openpose/experiment_0211/dataset/AUC_processed_split/"
    body_estimation = Body(MODEL_PATH)

    for camera in ['Camera 1', 'Camera 2']:
        for _, _, imgList in os.walk(os.path.join(ORIGIN_IMG_PATH, camera, dataSet, '')):
            imgTotal += len(imgList)

    for camera in ['Camera 1', 'Camera 2']:
        for dirPath, _, imgList in os.walk(os.path.join(ORIGIN_IMG_PATH, camera, dataSet, '')):
            for imgName in imgList:
                imgCount += 1
                print("{} dataset progress: {:5d}/{}".format(dataSet, imgCount, imgTotal))

                imgPath = os.path.join(dirPath, imgName)
                oriImg = cv2.imread(imgPath)
                rowList = skeleton_util.get_skeleton(body_estimation, oriImg)
                action_data.append(skeleton_util.compute_feature(rowList))
                action_label.append(int(dirPath[-1]))
    return (np.asarray(action_data), np.asarray(action_label))

def get_skeleton_data_by_csv(trainTestDataset, dataSet):
    action_data = []
    action_label = []

    f = open('../csv/' + dataSet + '/' + trainTestDataset + '_skeleton_output.csv','r',encoding = 'utf8')
    plots = csv.reader(f, delimiter=',')

    for i, rowList in enumerate(plots):
        if i != 0:
            action_data.append(skeleton_util.compute_feature(list(np.float_(rowList[2:len(rowList)]))))
            action_label.append(int(rowList[1]))
    f.close()
    return (np.asarray(action_data), np.asarray(action_label))

def get_skeleton_data(dataType, dataset):
    if dataType == "img":
        return get_skeleton_data_by_img("train"), get_skeleton_data_by_img("test")
    elif dataType == "csv":
        return get_skeleton_data_by_csv("train", dataset), get_skeleton_data_by_csv("test", dataset)

def train_with_skeleton_data(data_train, label_train):
    clf = RandomForestClassifier(criterion='entropy', n_estimators=150)
    clf.fit(data_train, label_train)
    return clf

if __name__ == '__main__':
    skeleton_data_start = time.time()
    (data_train, label_train), (data_test, label_test) = get_skeleton_data("csv", "AUC")
    skeleton_data_end = time.time()

    train_model_start = time.time()
    clf = train_with_skeleton_data(data_train, label_train)
    train_accuracy = clf.score(data_train, label_train)
    test_accuracy = clf.score(data_test, label_test)
    train_model_end = time.time()

    print()
    print("======================")
    print("get skeleton data time: {:.2f}".format(skeleton_data_end - skeleton_data_start))
    print("train model time: {:.2f}".format(train_model_end - train_model_start))
    print()
    print("train accuracy: {:.2%}".format(train_accuracy))
    print("test accuracy: {:.2%}".format(test_accuracy))
    print("======================")

    # predict_data = clf.predict_proba(data_test)