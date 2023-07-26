import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np
import csv
import time
import shutil
from tqdm import tqdm
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
sys.path.append("../..")
from src import model
from src import util
from src.body import Body
from src.hand import Hand

def get_skeleton(body_estimation, oriImg, class_img_name, data_class_name):
    candidate, subset = body_estimation(oriImg)
    if len(candidate) == 0:
        return

    [c_rows, c_cols] = candidate.shape
    [s_rows, s_cols] = subset.shape

    candidate_dict = {}
    for i in range(c_rows):
        candidate_dict[candidate[i, 3]] = [candidate[i, 0], candidate[i, 1]]

    skeletons_score_list = []
    for i in range(s_rows):
        skeletons_score_list.append(subset[i, s_cols-2])
    if (len(skeletons_score_list) == 0):
        return
    
    max_score_s_row = np.argmax(skeletons_score_list)
    rowList = []
    rowList.append(class_img_name)
    rowList.append(data_class_name)
    for j in range(s_cols):
        if j < s_cols-2:
            if subset[max_score_s_row, j] < 0:
                rowList.append(-1)
                rowList.append(-1)
            else:
                rowList.append(str(candidate_dict[subset[max_score_s_row, j]][0]))
                rowList.append(str(candidate_dict[subset[max_score_s_row, j]][1]))
        else:
            rowList.append(subset[max_score_s_row, j])
    return rowList, candidate, subset

def checkRemain(IMG_DIR_PATH, dataSet):
    remainImg = 0
    for _, _, imgList in os.walk(os.path.join(IMG_DIR_PATH, dataSet, '')):
        remainImg += len(imgList)
    return remainImg

if __name__ == '__main__':
    MODEL_PATH = "D:/Project/pytorch-openpose/model/body_pose_model.pth"
    IMG_DIR_PATH = "D:/Project/pytorch-openpose/experiment_0211/dataset/AUC_processed_merge_backup"
    # IMG_DIR_PATH = "D:/Project/pytorch-openpose/experiment_0211/dataset/StateFarm/backup_img"
    # OUTPUT_IMG_DIR = "D:/Project/pytorch-openpose/experiment_0211/dataset/AUC_processed_split_merge_output"

    start = time.time()
    body_estimation = Body(MODEL_PATH)
    breakThreshold = 20

    train_imgTotal = 0
    val_imgTotal = 0
    test_imgTotal = 0
    for dataSet in ['train', 'val', 'test']:
        for _, _, imgList in os.walk(os.path.join(IMG_DIR_PATH, dataSet, '')):
            if (dataSet == 'train'):
                train_imgTotal += len(imgList)
            elif (dataSet == 'test'):
                val_imgTotal += len(imgList)
            else:
                test_imgTotal += len(imgList)

    for dataSet in ['train', 'test']:
        datasetImgCount = 0
        CSV_PATH = "D:/Project/pytorch-openpose/experiment_0211/StateFarm_skeleton_" + (dataSet if dataSet != 'val' else 'train') + "_data.csv"
        while (checkRemain(IMG_DIR_PATH, dataSet) > 0):
            imgCount = 0
            canvasList = []
            rowList = []
            pbar = tqdm(total=breakThreshold)
            for dirPath, _, imgList in os.walk(os.path.join(IMG_DIR_PATH, dataSet, '')):
                for imgName in imgList:
                    imgPath = os.path.join(dirPath, imgName)
                    oriImg = cv2.imread(imgPath)
                    oriImg = cv2.resize(oriImg, (400, 400))
                    try:
                        rowData, candidate, subset = get_skeleton(body_estimation, oriImg, imgName, dirPath[-1])
                        datasetImgCount += 1
                        imgCount += 1
                    except TypeError:
                        shutil.move(imgPath, imgPath.replace("AUC_processed_merge_backup", "AUC_processed_merge_remove"))
                        # shutil.move(imgPath, imgPath.replace("backup_img", "remove_img"))
                        continue

                    pbar.update(1)
                    pbar.set_description("{} dataset | {} | dataset progress:  {:5d}/{}".format(
                        dataSet, "c"+dirPath[-1], datasetImgCount, train_imgTotal if dataSet == 'train' else (val_imgTotal if dataSet == 'val' else test_imgTotal)))
                    
                    rowList.append(rowData)
                    canvasList.append((imgPath, candidate, subset))
                    if (imgCount == breakThreshold): break
                if (imgCount == breakThreshold): break
            pbar.close()

            with open(CSV_PATH, 'a+', newline='') as csvfile:
                writer = csv.writer(csvfile)
                if (os.stat(CSV_PATH).st_size == 0):
                    writer.writerow(['img', 'class', 'Nose_x', 'Nose_y', 'Neck_x', 'Neck_y', 'RShoulder_x', 'RShoulder_y', 'RElbow_x', 'RElbow_y', 
                                    'RWrist_x', 'RWrist_y', 'LShoulder_x', 'LShoulder_y', 'LElbow_x', 'LElbow_y', 'LWrist_x', 'LWrist_y', 
                                    'RHip_x', 'RHip_y', 'RKnee_x', 'RKnee_y', 'RAnkle_x', 'RAnkle_y', 'LHip_x', 'LHip_y', 'LKnee_x', 
                                    'LKnee_y', 'LAnkle_x', 'LAnkle_y', 'REye_x', 'REye_y', 'LEye_x', 'LEye_y', 'REar_x', 'REar_y', 
                                    'LEar_x', 'LEar_y', 'total_score', 'total_parts'])
                    
                for idx, rowData in enumerate(rowList):
                    if len(rowData) > 0:
                        writer.writerow(rowData)
            
            for idx, canvasData in enumerate(canvasList):
                os.remove(canvasData[0])
            print("="*100)

            # pbar = tqdm(canvasList)
            # for idx, canvasData in enumerate(pbar):
            #     oriImg = cv2.imread(canvasData[0])
            #     canvas = copy.deepcopy(oriImg)
            #     canvas = util.draw_bodypose(canvas, canvasData[1], canvasData[2])
            #     plt.imshow(canvas[:, :, [2, 1, 0]])
            #     plt.axis('off')
            #     plt.savefig((canvasData[0]).replace("AUC_processed_split_merge_backup", "AUC_processed_split_merge_output"))
            #     plt.clf()
            #     os.remove(canvasData[0])
            #     pbar.set_description("canvas progress: ")
            # print("="*100)
    time_elapsed = time.time() - start
    print(f'get skeleton image and csv time: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print("="*100)