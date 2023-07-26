import joblib
import os
import cv2
import numpy as np
import skeleton_util
import time

import sys
sys.path.append("../..")
from src.body import Body

MODEL_NAME = ['Random Forest', 'AdaBoost', 'Gradient Boosting', 'XGBoost', 'Naive Bayes', 'Knn', 'Decision Tree', 'Ensemble Stacking']
# MODEL_NAME = ['Random Forest', 'AdaBoost', 'Gradient Boosting', 'XGBoost', 'Naive Bayes', 'Knn', 'Decision Tree', 'Ensemble Stacking']

BODY_MODEL_PATH = "../../model/body_pose_model.pth"
CLASS_IMG_DIR_PATH = '../test_img/'

body_estimation = Body(BODY_MODEL_PATH)

print("========================")
for model_name in MODEL_NAME:
    CLASSIFIER_MODEL_PATH = '../model/sklearn_classifier/' + model_name + '_model.pkl'
    clf = joblib.load(CLASSIFIER_MODEL_PATH)

    print("model: " + model_name)

    prediction_time = []
    for class_img_name in os.listdir(CLASS_IMG_DIR_PATH):
        detection_image = CLASS_IMG_DIR_PATH + class_img_name

        start = time.time()
        oriImg = cv2.imread(detection_image)

        try:
            rowList, candidate, subset = skeleton_util.get_skeleton(body_estimation, oriImg, class_img_name, class_img_name[0])
            prediction = clf.predict(np.array([rowList[2:len(rowList)]], dtype=object))[0]
            # print("actual: " + str(class_img_name[0]) + " , prediction: " + str(prediction))

            end = time.time()
            prediction_time.append(end - start)
        except TypeError:
            print("can't find skeleton!")

    if (len(prediction_time) > 0):
        print("average prediction time: {:.2f}s".format(np.mean(prediction_time)))
    print("========================")