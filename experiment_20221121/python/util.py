from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from datetime import datetime
import csv
import os

def get_data_and_label():
    action_data = []
    action_label = []

    # csv_fileName = "../final_skeleton_output_14_arm.csv"
    # csv_fileName = "../final_skeleton_output_10.csv"
    csv_fileName = "../csv/merge_skeleton_output.csv"
    f = open(csv_fileName,'r',encoding = 'utf8')
    plots = csv.reader(f, delimiter=',')

    for i, item in enumerate(plots):
        if i != 0:
            row = []
            for j in range(len(item)):
                if j != 0 and j != 1:
                    row.append(float(item[j]))
                elif j == 1:
                    action_label.append(int(item[j]))
            action_data.append(row)
    f.close()

    return shuffle_data(action_data, action_label)

def get_np_asarray_data_and_label():
    action_data, action_label = get_data_and_label()
    return np.asarray(action_data), np.asarray(action_label)

def shuffle_data(action_data, action_label):
    index_list = [i for i in range(len(action_data))]
    np.random.shuffle(index_list)

    shuffle_data = []
    shuffle_label = []
    for index in index_list:
        shuffle_data.append(action_data[index])
        shuffle_label.append(action_label[index])
    return shuffle_data, shuffle_label
## ======================================================== ##
def pca_for_action_data(action_data):
    PCA_COMPONENTS = 19

    pca = PCA(n_components=PCA_COMPONENTS)
    pca.fit(action_data)
    action_data_reduced = pca.transform(action_data)
    return action_data_reduced

def lda_for_action_data(action_data, action_label):
    LDA_COMPONENTS = 19

    lda = LinearDiscriminantAnalysis(n_components=LDA_COMPONENTS)
    lda.fit(action_data, action_label)
    action_data_reduced = lda.transform(action_data)
    return action_data_reduced
## ======================================================== ##
def get_time():
    return datetime.now().strftime("%Y%m%d_%H%M")
## ======================================================== ##
def remove_specific_class_and_feature_outliers(action_data, action_label, cls, index):
    specific_cls_and_feature_data_without_nan = []
    for i, data in enumerate(action_data):
        if (action_label[i] == cls and data[index] > 0):
            specific_cls_and_feature_data_without_nan.append(data[index])
    
    IQ = np.percentile(specific_cls_and_feature_data_without_nan, [25, 75])
    IQR = IQ[1] - IQ[0]

    remove_outlier_data = []
    remove_outlier_label = []
    for i, data in enumerate(action_data):
        if ((action_label[i] != cls) or ((data[index] < IQ[1]+1.5*IQR and data[index] > IQ[0]-1.5*IQR) or data[index] == -1)):
            remove_outlier_data.append(data)
            remove_outlier_label.append(action_label[i])
    return remove_outlier_data, remove_outlier_label