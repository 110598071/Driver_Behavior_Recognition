import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np
import csv
import shutil
import skeleton_util
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
sys.path.append("../..")
from src import model
from src import util
from src.body import Body
from src.hand import Hand

MODEL_PATH = "../../model/body_pose_model.pth"
IMG_DIR_PATH = "D:/Project/data/driver distraction/ggg/"
OUTPUT_IMG_DIR = "../output_img/"
ITERATION = 0
MAX_ITERATION = 200
RENEW_MODEL = 20

body_estimation = Body(MODEL_PATH)

data_class_name_list = []
for inner_dir_name in os.listdir(IMG_DIR_PATH):
    data_class_name_list.append(inner_dir_name)

# for origin_output_dir in os.listdir(OUTPUT_IMG_DIR):
#     shutil.rmtree(OUTPUT_IMG_DIR + origin_output_dir)

for data_class_name in data_class_name_list:
    CLASS_IMG_DIR_PATH = IMG_DIR_PATH  + data_class_name + "/"
    OUTPUT_IMG_DIR_PATH = OUTPUT_IMG_DIR + data_class_name + "/"
    FINAL_SKELETON_CSVPATH = "../csv/" + data_class_name + "_skeleton_output.csv"

    class_img_amount = len(os.listdir(CLASS_IMG_DIR_PATH))
    count = 0

    # if os.path.exists(FINAL_SKELETON_CSVPATH):
    #     os.remove(FINAL_SKELETON_CSVPATH)

    with open(FINAL_SKELETON_CSVPATH, 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if (os.stat(FINAL_SKELETON_CSVPATH).st_size == 0):
            writer.writerow(['img', 'class', 'Nose_x', 'Nose_y', 'Neck_x', 'Neck_y', 'RShoulder_x', 'RShoulder_y', 'RElbow_x', 'RElbow_y', 
                            'RWrist_x', 'RWrist_y', 'LShoulder_x', 'LShoulder_y', 'LElbow_x', 'LElbow_y', 'LWrist_x', 'LWrist_y', 
                            'RHip_x', 'RHip_y', 'RKnee_x', 'RKnee_y', 'RAnkle_x', 'RAnkle_y', 'LHip_x', 'LHip_y', 'LKnee_x', 
                            'LKnee_y', 'LAnkle_x', 'LAnkle_y', 'REye_x', 'REye_y', 'LEye_x', 'LEye_y', 'REar_x', 'REar_y', 
                            'LEar_x', 'LEar_y', 'total_score', 'total_parts'])

        for class_img_name in os.listdir(CLASS_IMG_DIR_PATH):
            if (ITERATION % RENEW_MODEL == 0):
                print("renew model!!!")
                body_estimation = Body(MODEL_PATH)

            ITERATION += 1
            count += 1
            detection_image = CLASS_IMG_DIR_PATH + class_img_name
            print(detection_image + "  " + str(count) + "/" + str(class_img_amount) + "  total iteration: " + str(ITERATION) + "/" + str(MAX_ITERATION))

            oriImg = cv2.imread(detection_image)
            try:
                rowList, candidate, subset = skeleton_util.get_skeleton(body_estimation, oriImg, class_img_name, data_class_name[1])
            except TypeError:
                continue

            if len(rowList) > 0:
                writer.writerow(rowList)

            canvas = copy.deepcopy(oriImg)
            canvas = util.draw_bodypose(canvas, candidate, subset)

            plt.imshow(canvas[:, :, [2, 1, 0]])
            plt.axis('off')

            os.remove(detection_image)
            if os.path.isdir(OUTPUT_IMG_DIR_PATH):
                pass
            else:
                os.makedirs(OUTPUT_IMG_DIR_PATH)
            plt.savefig(OUTPUT_IMG_DIR_PATH  + class_img_name)

            if (ITERATION >= MAX_ITERATION): break
    if (ITERATION >= MAX_ITERATION): break