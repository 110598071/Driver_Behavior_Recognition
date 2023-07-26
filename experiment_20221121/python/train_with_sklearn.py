from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn import tree
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import csv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import joblib
import time
import util
import feature_data
import hyperparameter_util
import hyperparameter_optimization

#========================================================================#
# all_models_name = ['Random Forest', 'AdaBoost', 'Gradient Boosting', 'XGBoost', 'Naive Bayes', 'Knn', 'Decision Tree', 'Ensemble Stacking']
all_models_name = ['Random Forest', 'XGBoost', 'Knn', 'Ensemble Stacking']

# models_name = ['Random Forest', 'XGBoost', 'Knn', 'Ensemble Stacking']
# models_name = ['Random Forest']
# models_name = ['XGBoost']
# models_name = ['Knn']
# models_name = ['Ensemble Stacking']
models_name = ['Random Forest', 'XGBoost', 'Knn']
#========================================================================#
HYPERPARAMETER_OPTIMIZATION = False
TRAIN_DEFAULT = True

SAVE_MODEL = False
STORE_CSV = True
STORE_REPORT = False

DRAW_MERGE_MATRIX = False
DRAW_INDEPENDENT_MATRIX = False
DRAW_PERCENTAGE_MATRIX = True

DATA_STANDARIZATION = False
REMOVE_OUTLIERS = False
#========================================================================#
# all_models_dict = {
#     'Random Forest': RandomForestClassifier(),
#     'AdaBoost': AdaBoostClassifier(),
#     'Gradient Boosting': GradientBoostingClassifier(),
#     'XGBoost': XGBClassifier(),
#     'Naive Bayes': GaussianNB(),
#     'Knn': KNeighborsClassifier(),
#     'Decision Tree': tree.DecisionTreeClassifier(),
#     'Ensemble Stacking': StackingClassifier(estimators=estimators),
# }
#========================================================================#
OUTLIERS_CLS = [8]
OUTLIERS_INDEX = [[6, 8, 16, 17]]

MODEL_PATH = '../model/sklearn_classifier/'
SKLEARN_MODELS_ACCURACY_CSV_PATH = "../sklearn_accuracy/sklearn_accuracy.csv"
SKLEARN_MODELS_ACCURACY_EXCEL_PATH = "../sklearn_accuracy/sklearn_accuracy.xlsx"
SKLEARN_MODELS_CLASSIFICATION_REPORT_CSV_PATH = "../sklearn_accuracy/sklearn_classification_report.csv"
SKLEARN_MODELS_CLASSIFICATION_REPORT_EXCEL_PATH = "../sklearn_accuracy/sklearn_classification_report.xlsx"
K_FOLD = 10
SUBPLOT_COL = 3
SUBPLOT_ROW = int(np.ceil(len(models_name) / SUBPLOT_COL))
TIME = util.get_time()
#========================================================================#

def build_optimized_models_dict():
    all_models_dict = {
        'Random Forest': RandomForestClassifier(**hyperparameter_util.random_forest_params()),
        'XGBoost': XGBClassifier(**hyperparameter_util.xgboost_params()),
        'Knn': KNeighborsClassifier(**hyperparameter_util.knn_params()),
        'Ensemble Stacking': StackingClassifier(**hyperparameter_util.ensemble_stacking_with_LR_params()),
    }
    return all_models_dict

def build_default_models_dict():
    estimators = [
        ('RF', RandomForestClassifier()),
        ('XGB', XGBClassifier()),
        ('KNN', KNeighborsClassifier())
    ]

    all_models_dict = {
        'Random Forest': RandomForestClassifier(),
        'XGBoost': XGBClassifier(),
        'Knn': KNeighborsClassifier(),
        'Ensemble Stacking': StackingClassifier(estimators=estimators),
    }
    return all_models_dict

def get_models_dict(idx):
    if idx == 0:
        all_models_dict = build_default_models_dict()
    elif idx == 1:
        all_models_dict = build_optimized_models_dict()

    models = []
    for name in models_name:
        models.append(all_models_dict[name])

    models_dict = dict(zip(models_name, models))
    return models_name, models_dict
#========================================================================#
def train_and_get_accuracy(idx, data_train, label_train, data_test, label_test):
    subplot_index = 1
    optimization_info = ['', '', '']
    merge_matrix_list = []
    models_accuracy_and_time = []
    actual_label = []
    prediction_label = []
    models_name, models_dict = get_models_dict(idx)
    print("========================")
    for name in models_name:
        print('model: ' + name)
        print('params: ' + 'default' if idx==0 else 'optimized')
        print()

        matrix_list = [[0]*10 for _ in range(10)]

        clf = models_dict[name]
        if idx == 1 and HYPERPARAMETER_OPTIMIZATION:
            params, optimization_info = hyperparameter_optimization.get_hyperparameter_optimization(name, data_train, label_train, DATA_STANDARIZATION, REMOVE_OUTLIERS, OUTLIERS_CLS, OUTLIERS_INDEX)
            clf.set_params(**params)

        start = time.time()
        clf.fit(data_train, label_train)

        train_accuracy = clf.score(data_train, label_train)
        test_accuracy = clf.score(data_test, label_test)

        if SAVE_MODEL:
            joblib.dump(clf, MODEL_PATH + name + '/' + name + '_model_' + TIME + judge_idx(idx) + '.pkl')

        prediction_list = clf.predict(data_test)
        for i, predict in enumerate(prediction_list):
            matrix_list[label_test[i]][predict] += 1
        merge_matrix_list.append(matrix_list)
        actual_label.append(label_test)
        prediction_label.append(prediction_list)
        draw_confusion_matrix(matrix_list, name, False, subplot_index)

        end = time.time()
        models_accuracy_and_time.append("{:.2%}".format(train_accuracy))
        models_accuracy_and_time.append("{:.2%}".format(test_accuracy))
        models_accuracy_and_time.append("{:.2f}".format(end - start))

        print(name + " train accuray: {:.2%}".format(train_accuracy))
        print(name + " test accuray: {:.2%}".format(test_accuracy))
        print("========================")
        subplot_index += 1

    return merge_matrix_list, models_accuracy_and_time, (actual_label, prediction_label), optimization_info
#========================================================================#
def store_model_accuracy_to_csv(models_accuracy_and_time, optimization_info):
    accuracy_and_time = [TIME]
    model_list_len = len(models_name)
    name_index = 0
    accuracy_index = 0

    default_accuracy, optimized_accuracy = models_accuracy_and_time[0], models_accuracy_and_time[1]
    for name in all_models_name:
        if (name_index == model_list_len or name != models_name[name_index]):
            accuracy_and_time.extend(['', '', '', '', '', ''])
        else:
            accuracy_and_time.extend(default_accuracy[accuracy_index : accuracy_index+3] if TRAIN_DEFAULT else ['', '', ''])
            accuracy_and_time.extend(optimized_accuracy[accuracy_index : accuracy_index+3])
            accuracy_index += 3
            name_index += 1
    accuracy_and_time.extend(optimization_info)

    with open(SKLEARN_MODELS_ACCURACY_CSV_PATH, 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(accuracy_and_time)

    cvsDataframe = pd.read_csv(SKLEARN_MODELS_ACCURACY_CSV_PATH)
    resultExcelFile = pd.ExcelWriter(SKLEARN_MODELS_ACCURACY_EXCEL_PATH)
    cvsDataframe.to_excel(resultExcelFile, index=False)
    resultExcelFile.save()

def store_classification_report(idx, report_label):
    (actual_label, prediction_label) = report_label
    report_list = []
    for i, name in enumerate(models_name):
        report_dict = classification_report(actual_label[i], prediction_label[i], digits=4, output_dict=True)
        report_list.append([name, '', '', '', ''])
        report_list.append(['default' if idx==0 else 'optimized', 'precision', 'recall', 'f1-score', 'support'])

        for i in range(10):
            report_row = np.round(list(report_dict.get(str(i)).values()), 4).tolist()
            report_row.insert(0, str(i))
            report_list.append(report_row)
        macro_row = np.round(list(report_dict.get('macro avg').values()), 4).tolist()
        macro_row.insert(0, 'macro avg')
        weighted_row = np.round(list(report_dict.get('weighted avg').values()), 4).tolist()
        weighted_row.insert(0, 'weighted avg')

        report_list.append(macro_row)
        report_list.append(weighted_row)
        report_list.append(['accuracy', '', '', np.round(report_dict.get('accuracy'), 4), ''])
        report_list.append(['', '', '', '', ''])

    if idx == 0:
        with open(SKLEARN_MODELS_CLASSIFICATION_REPORT_CSV_PATH, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in report_list:
                writer.writerow(row)
    elif idx == 1:
        with open(SKLEARN_MODELS_CLASSIFICATION_REPORT_CSV_PATH, 'a+', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in report_list:
                writer.writerow(row)

    cvsDataframe = pd.read_csv(SKLEARN_MODELS_CLASSIFICATION_REPORT_CSV_PATH)
    resultExcelFile = pd.ExcelWriter(SKLEARN_MODELS_CLASSIFICATION_REPORT_EXCEL_PATH)
    cvsDataframe.to_excel(resultExcelFile, index=False)
    resultExcelFile.save()
#========================================================================#
def create_subplots_and_adjust_figsize():
    plt.figure()
    fig, axs = plt.subplots(SUBPLOT_ROW, SUBPLOT_COL, figsize=(30-5*(SUBPLOT_ROW-1), 14) if SUBPLOT_ROW>1 else (20, 7))
    fig.tight_layout(pad=3.0)

    empty_plot_amount = SUBPLOT_COL * SUBPLOT_ROW - len(models_name)

    for i in range(-1, -1-empty_plot_amount, -1):
        if (SUBPLOT_ROW == 1):
            axs[i].axis('off')
        else:
            axs[-1, i].axis('off')

def draw_confusion_matrix(matrix_list, model_name, draw_independent, subplot_index=0):
    if DRAW_PERCENTAGE_MATRIX:
        for i, row in enumerate(matrix_list):
            row_sum = np.sum(row)
            for j, num in enumerate(row):
                matrix_list[i][j] = np.round(num/row_sum, 2)

    ticks = [list(range(10))]
    df = pd.DataFrame(matrix_list, columns=ticks, index=ticks)

    if draw_independent:
        plt.figure()
    else:
        plt.subplot(SUBPLOT_ROW, SUBPLOT_COL, subplot_index)
    sns.heatmap(df, fmt='', annot=True)
    plt.title(model_name + ' confusion_matrix')
    plt.ylabel('actual label')
    plt.xlabel('predicted label')

def show_and_save_merge_confusion_matrix(idx):
    HEATMAP_PATH = '../heatmap/sklearn_classifier/merge/clf_heatmap_' + TIME + judge_idx(idx) +'.png'
    plt.savefig(HEATMAP_PATH, dpi=300)

def draw_and_save_independent_confusion_matrix(idx, merge_matrix_list):
    HEATMAP_PATH = '../heatmap/sklearn_classifier/'
    for i, model_name in enumerate(models_name):
        INDEPENDENT_HEATMAP_PATH = HEATMAP_PATH + model_name + "/clf_heatmap_" + TIME + judge_idx(idx) +'.png'
        draw_confusion_matrix(merge_matrix_list[i], model_name, True)
        plt.savefig(INDEPENDENT_HEATMAP_PATH, dpi=300)
#========================================================================#
def judge_idx(idx):
    if idx == 0:
        return '_default'
    elif idx == 1:
        return '_optimized'

def train_models():
    merge_accuracy = []
    (data_train, label_train), (data_test, label_test) = feature_data.get_train_and_test_data()
    for idx in range(2):
        if (not TRAIN_DEFAULT) and idx == 0: 
            merge_accuracy.append([])
            continue

        create_subplots_and_adjust_figsize()
        merge_matrix_list, models_accuracy_and_time, report_label, optimization_info = train_and_get_accuracy(idx, data_train, label_train, data_test, label_test)
        merge_accuracy.append(models_accuracy_and_time)
        if STORE_REPORT: store_classification_report(idx, report_label)
        if DRAW_MERGE_MATRIX: show_and_save_merge_confusion_matrix(idx)
        if DRAW_INDEPENDENT_MATRIX: draw_and_save_independent_confusion_matrix(idx, merge_matrix_list)
    if STORE_CSV: store_model_accuracy_to_csv(merge_accuracy, optimization_info)
#========================================================================#
if __name__ == '__main__':
    train_models()