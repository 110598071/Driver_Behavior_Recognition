from .config import NUM_CLASSES, DEVICE, INFERENCE_MODEL, RESIZE_TO, IOU_THRESHOLD_UPPER, IOU_THRESHOLD_LOWER, IOU_THRESHOLD_STEP, VALID_DIR, OBJECT_AMOUNT_LIMIT
from .model import create_model
from .inference import inference
from .feature_computation import compute_overlap_area, compute_box_area
import torch
import numpy as np
from xml.etree import ElementTree as et
import math
import cv2
import os
import glob
from tqdm import tqdm
from collections import namedtuple
Rectangle = namedtuple('Rectangle', 'x_min, y_min, x_max, y_max')
Point = namedtuple('Point', 'x, y')

class ObjectIoUComputation:
    def __init__(self, boxes, pred_cls, scores, imgPath):
        self.boxes = boxes
        self.pred_cls = pred_cls
        self.scores = scores
        self.imgPath = imgPath
        self.IoU_list = {}
        
        self.ground_true_boxs = []
        self.ground_true_labels = []
        self.get_ground_true_box_and_label()
        for label in self.ground_true_labels:
            self.IoU_list[label] = 0.0

        self.arrange_object()
        self.handle_IoU_list()

    def find_boxes(self, target_label):
        find_boxes_list = []
        for idx, box in enumerate(self.boxes):
            if (self.pred_cls[idx] == target_label):
                find_boxes_list.append(box)
        return find_boxes_list

    def arrange_object(self):
        for key, value in OBJECT_AMOUNT_LIMIT.items():
            if len(self.find_boxes(key)) > value:
                for _ in range(len(self.find_boxes(key)) - value):
                    self.remove_object(key)

    def remove_object(self, target_label):
        occurrence = [idx for idx, cls in enumerate(self.pred_cls) if cls == target_label]
        del self.boxes[occurrence[-1]]
        del self.pred_cls[occurrence[-1]]
        del self.scores[occurrence[-1]]

    def get_ground_true_box_and_label(self):
        annot_file_path = self.imgPath.replace('.jpg', '.xml')
        tree = et.parse(annot_file_path)
        root = tree.getroot()

        image = cv2.imread(self.imgPath)
        image_width = image.shape[1]
        image_height = image.shape[0]
        
        for member in root.findall('object'):
            self.ground_true_labels.append(member.find('name').text)
            
            xmin = int(member.find('bndbox').find('xmin').text)
            xmax = int(member.find('bndbox').find('xmax').text)
            ymin = int(member.find('bndbox').find('ymin').text)
            ymax = int(member.find('bndbox').find('ymax').text)
            
            xmin_final = int((xmin/image_width)*RESIZE_TO)
            xmax_final = int((xmax/image_width)*RESIZE_TO)
            ymin_final = int((ymin/image_height)*RESIZE_TO)
            yamx_final = int((ymax/image_height)*RESIZE_TO)
            
            self.ground_true_boxs.append([xmin_final, ymin_final, xmax_final, yamx_final])

    def handle_IoU_list(self):
        IoU = 0.0
        hand_IoU = []
        for idx, gt_label in enumerate(self.ground_true_labels):
            pred_boxes_list = self.find_boxes(gt_label)
            if len(pred_boxes_list) == 2:
                compute_IoU_list = []
                for pred_box in pred_boxes_list:
                    compute_IoU_list.append(self.compute_IoU(self.ground_true_boxs[idx], pred_box))
                
                if self.ground_true_labels.count(gt_label) == 1:
                    IoU = np.max(compute_IoU_list)
                else:
                    hand_IoU.append(compute_IoU_list)
                    if len(hand_IoU) == 2:
                        opt1 = np.round((hand_IoU[0][0] +hand_IoU[1][1]) / 2, 3)
                        opt2 = np.round((hand_IoU[0][1] +hand_IoU[1][0]) / 2, 3)
                        IoU = np.max([opt1, opt2])
            elif len(pred_boxes_list) == 1:
                IoU = self.compute_IoU(self.ground_true_boxs[idx], pred_boxes_list[0])
            self.IoU_list[gt_label] = IoU

    def compute_IoU(self, gt_box, pred_box):
        gt_rectangle = Rectangle._make(gt_box)
        pred_rectangle = Rectangle._make(pred_box)
        overlap_area = compute_overlap_area(gt_rectangle, pred_rectangle)
        union_area = compute_box_area(gt_rectangle) + compute_box_area(pred_rectangle) - overlap_area
        return np.round(overlap_area / union_area, 3)
    
    def get_IoU_list(self):
        return self.IoU_list

    def print_object_info(self):
        print(f'boxes: {self.boxes}')
        print(f'pred_cls: {self.pred_cls}')
        print(f'scores: {self.scores}')
        print(f'ground_true_boxs: {self.ground_true_boxs}')
        print(f'ground_true_labels: {self.ground_true_labels}')
        print(f'IoU_list: {self.IoU_list}')

def IoU_computation(model):
    imgPathList = []
    for subdir in os.listdir(VALID_DIR):
        imgPathList.extend(glob.glob(f"{VALID_DIR}/{subdir}/*.jpg"))

    final_IoU_list = []
    avg_IoU_list = []
    threshold_list = np.arange(IOU_THRESHOLD_LOWER, IOU_THRESHOLD_UPPER, IOU_THRESHOLD_STEP)
    threshold_list = [np.round(threshold, 3) for threshold in threshold_list]
    for detection_threshold in threshold_list:
        IoU_combine_dict = {}
        count_dict = {}
        for key in OBJECT_AMOUNT_LIMIT.keys():
            IoU_combine_dict[key] = 0.0
            count_dict[key] = 0

        print('='*30)
        print(f'detection threshold: {detection_threshold}')
        boxes_list, pred_cls_list, scores_list = inference(model, imgPathList, False, detection_threshold)
        for idx, boxes in enumerate(boxes_list):
            objectIoUComputation = ObjectIoUComputation(boxes, pred_cls_list[idx], scores_list[idx], imgPathList[idx])
            # objectIoUComputation.print_object_info()
            IoU_list = objectIoUComputation.get_IoU_list()

            for key, value in IoU_list.items():
                IoU_combine_dict[key] += value
                count_dict[key] += 1
        
        iter_IoU_dict = {}
        for key, value in IoU_combine_dict.items():
            if value == 0.0:
                iter_IoU_dict[key] = 0.0
            else:
                iter_IoU_dict[key] = np.round(value / count_dict[key], 3)
        IoU_total = 0.0
        for value in iter_IoU_dict.values():
            IoU_total += value
        iter_IoU_dict['average'] = np.round(IoU_total / len(IoU_combine_dict), 3)

        final_IoU_list.append(iter_IoU_dict)
        avg_IoU_list.append(iter_IoU_dict['average'])
    return final_IoU_list[np.argmax(avg_IoU_list)], threshold_list[np.argmax(avg_IoU_list)]