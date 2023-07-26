import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np
import torch
import time
import threading
import joblib

import skeleton_util

import sys
sys.path.append("../..")
from src import util
from src.body import Body

MODEL_NAME = 'Gradient Boosting'
CLASSIFIER_MODEL_PATH = '../model/sklearn_classifier/' + MODEL_NAME + '_model.pkl'
BODY_MODEL_PATH = "../../model/body_pose_model.pth"

clf = joblib.load(CLASSIFIER_MODEL_PATH)
body_estimation = Body(BODY_MODEL_PATH)

print(f"Torch device: {torch.cuda.get_device_name()}")

class ipcamCapture:
    def __init__(self, index):
        self.Frame = []
        self.status = False
        self.isstop = False
		
        self.capture = cv2.VideoCapture(index)
        self.capture.set(3, 1280)
        self.capture.set(4, 960)
        self.capture.set(cv2.CAP_PROP_FPS, 30)

    def start(self):
        print('ipcam started!')
        threading.Thread(target=self.queryframe, daemon=True, args=()).start()

    def stop(self):
        self.isstop = True
        print('ipcam stopped!')
   
    def getframe(self):
        return self.status, self.Frame.copy()
        
    def queryframe(self):
        while (not self.isstop):
            self.status, self.Frame = self.capture.read()
        self.capture.release()

ipcam = ipcamCapture(0)
ipcam.start()
time.sleep(1)

while True:
    ret, oriImg = ipcam.getframe()
    candidate, subset = body_estimation(oriImg)
    canvas = copy.deepcopy(oriImg)
    canvas = util.draw_bodypose(canvas, candidate, subset)

    rowList, candidate, subset = skeleton_util.get_skeleton(body_estimation, oriImg, "img_name", "img_class")
    [proba_list] = clf.predict_proba(np.array([rowList[2:len(rowList)]], dtype=object))
    prediction = np.argmax(proba_list)

    for i in range(10):
        if (i == prediction):
            color = (0, 255, 255)
        else:
            color = (34, 139, 34)
        cv2.putText(canvas, ("class" + str(i) + ": {:.2f}%").format(proba_list[i]*100), (20, 40+30*i), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 1, cv2.LINE_AA)

    cv2.imshow('demo', canvas)

    if cv2.waitKey(1000) == 27:
        cv2.destroyAllWindows()
        ipcam.stop()
        break