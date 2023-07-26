from .config import CLASSES, RESIZE_TO
from .model import create_model
from .utils import arrange_object
from .feature_computation import compute_overlap_area, compute_box_area
import numpy as np
import cv2
import os
import torch
import matplotlib.pyplot as plt
from xml.etree import ElementTree as et
import glob as glob
import tqdm
from collections import namedtuple
Rectangle = namedtuple('Rectangle', 'x_min, y_min, x_max, y_max')

MAP_THRESHOLD = 0.5

def mAP_inference(model, test_images):
    boxes_list = []
    pred_cls_list = []
    scores_list = []
    pbar = tqdm.tqdm(range(len(test_images)))
    for i in pbar:
        image_name = test_images[i].split('\\')[-1].split('.')[0]
        image = cv2.imread(test_images[i])
        orig_image = image.copy()
        orig_image = cv2.resize(orig_image, (RESIZE_TO, RESIZE_TO))

        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = cv2.resize(image, (RESIZE_TO, RESIZE_TO))
        image /= 255.0
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # image = np.stack((image,)*3, axis=-1)

        # bring color channels to front
        image = np.transpose(image, (2, 0, 1)).astype(np.float64)
        image = torch.tensor(image, dtype=torch.float).cuda()
        image = torch.unsqueeze(image, 0)
        with torch.no_grad():
            outputs = model(image)
        
        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
        if len(outputs[0]['boxes']) != 0:
            boxes = outputs[0]['boxes'].data.numpy()
            scores = outputs[0]['scores'].data.numpy()
            boxes = boxes.astype(np.int32)
            pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]

            boxes_list.append(boxes)
            pred_cls_list.append(pred_classes)
            scores_list.append(scores)

        pbar.set_description('Fast RCNN inference')
    return boxes_list, pred_cls_list, scores_list

def get_ground_true_boxes(test_images):
    ground_true_boxes_list = []
    ground_true_cls_list = []

    for image_path in test_images:
        image = cv2.imread(image_path)
        annot_file_path = image_path.replace('.jpg', '.xml')
            
        boxes = []
        labels = []
        tree = et.parse(annot_file_path)
        root = tree.getroot()
        
        for member in root.findall('object'):
            labels.append(member.find('name').text)
            
            xmin = int(member.find('bndbox').find('xmin').text)
            xmax = int(member.find('bndbox').find('xmax').text)
            ymin = int(member.find('bndbox').find('ymin').text)
            ymax = int(member.find('bndbox').find('ymax').text)
            
            image_width = image.shape[1]
            image_height = image.shape[0]

            xmin_final = (xmin/image_width)*RESIZE_TO
            xmax_final = (xmax/image_width)*RESIZE_TO
            ymin_final = (ymin/image_height)*RESIZE_TO
            yamx_final = (ymax/image_height)*RESIZE_TO
            boxes.append([xmin_final, ymin_final, xmax_final, yamx_final])

        ground_true_boxes_list.append(boxes)
        ground_true_cls_list.append(labels)
    return ground_true_boxes_list, ground_true_cls_list

def compute_IoU(gt_box, pred_box):
    gt_rectangle = Rectangle._make(gt_box)
    pred_rectangle = Rectangle._make(pred_box)
    overlap_area = compute_overlap_area(gt_rectangle, pred_rectangle)
    union_area = compute_box_area(gt_rectangle) + compute_box_area(pred_rectangle) - overlap_area
    return np.round(overlap_area / union_area, 3) if union_area != 0 else 0.0

def check_overlap(gt_box, pred_box):
    gt_rectangle = Rectangle._make(gt_box)
    pred_rectangle = Rectangle._make(pred_box)
    overlap_area = compute_overlap_area(gt_rectangle, pred_rectangle)
    return overlap_area > (0.6 * compute_box_area(gt_rectangle)) or overlap_area > (0.6 * compute_box_area(pred_rectangle))

def compute_mAP(pred_boxes_list, pred_cls_list, pred_scores_list, ground_true_boxes_list, ground_true_cls_list):
    AP_list = []
    precision_list = []
    recall_list = []
    cls_list = []
    mAP = 0

    for cls in CLASSES[1:len(CLASSES)]:
    # for cls in ['hand']:
        img_AP_list = []
        img_precision_list = []
        img_recall_list = []

        for img_idx in range(len(ground_true_cls_list)):
            actual_positive_amount = 0
            gt_img_cls_boxes_list = []
            for gt_idx, gt_cls in enumerate(ground_true_cls_list[img_idx]):
                if gt_cls == cls:
                    actual_positive_amount += 1
                    gt_img_cls_boxes_list.append(ground_true_boxes_list[img_idx][gt_idx])

            if actual_positive_amount == 0:
                continue

            img_score_list = []
            img_IoU_list = []

            if len(pred_cls_list[img_idx]) == 0:
                img_AP, img_precision, img_recall = 0.0, 0.0, 0.0
            else:
                arrange_img_pred_boxes_list, arrange_img_pred_scores_list = arrange_img_boxes(pred_boxes_list[img_idx], pred_cls_list[img_idx], pred_scores_list[img_idx], cls)
                if len(arrange_img_pred_boxes_list) == 0:
                    img_AP, img_precision, img_recall = 0.0, 0.0, 0.0
                else:
                    for pred_obj_idx, img_pred_score in enumerate(arrange_img_pred_scores_list):
                        img_score_list.append(img_pred_score)
                        img_IoU_list.append(np.max([compute_IoU(gt_img_cls_box, arrange_img_pred_boxes_list[pred_obj_idx].tolist()) for gt_img_cls_box in gt_img_cls_boxes_list]))
                    img_AP, img_precision, img_recall = compute_AP(img_score_list, img_IoU_list, actual_positive_amount)

            img_AP_list.append(img_AP)
            img_precision_list.append(img_precision)
            img_recall_list.append(img_recall)

        if len(img_AP_list) == 0:
            continue

        # print(img_AP_list)
        # print(img_precision_list)
        # print(img_recall_list)

        AP_list.append(np.round(np.mean(img_AP_list), 4))
        precision_list.append(np.round(np.mean(img_precision_list), 4))
        recall_list.append(np.round(np.mean(img_recall_list), 4))
        cls_list.append(cls)
    mAP = np.round(np.mean(AP_list), 4)
    return mAP, precision_list, recall_list, cls_list, AP_list

def compute_AP(img_score_list, img_IoU_list, actual_positive_amount):
    zipped = zip(img_score_list, img_IoU_list)
    zipped = sorted(zipped, reverse=True)
    img_score_list, img_IoU_list = zip(*zipped)

    precision_record_list = []
    recall_record_list = []

    true_positive_amount = 0
    predict_positive_amount = 0
    for img_IoU in img_IoU_list:
        predict_positive_amount += 1
        if img_IoU >= MAP_THRESHOLD:
            true_positive_amount += 1
        precision_record_list.append(np.round((true_positive_amount / predict_positive_amount), 3))
        recall_record_list.append(np.round((true_positive_amount / actual_positive_amount), 3))

    # print(precision_record_list)
    # print(recall_record_list)
    # print()

    AP = 0.0
    if predict_positive_amount == 1:
        AP = precision_record_list[0] * recall_record_list[0]
    else:
        for idx in range(0, len(precision_record_list)):
            if idx == 0:
                AP += recall_record_list[idx] * precision_record_list[idx]
            else:
                AP += (recall_record_list[idx] - recall_record_list[idx-1]) * np.max([precision_record_list[idx], precision_record_list[idx-1]])
    return AP, precision_record_list[-1], recall_record_list[-1]

def arrange_img_boxes(img_pred_boxes_list, img_pred_cls_list, img_pred_scores_list, cls):
    cls_img_pred_boxes_list = []
    cls_img_pred_scores_list = []

    for idx, pred_cls in enumerate(img_pred_cls_list):
        if pred_cls == cls:
            cls_img_pred_boxes_list.append(img_pred_boxes_list[idx])
            cls_img_pred_scores_list.append(img_pred_scores_list[idx])

    if len(cls_img_pred_scores_list) == 0:
        return [], []
    
    cls_img_pred_boxes_list.reverse()
    cls_img_pred_scores_list.reverse()
    arrange_img_pred_boxes_list = []
    arrange_img_pred_scores_list = []

    for idx in range(0, len(cls_img_pred_boxes_list)-1):
        reserve = True
        for fowrard_idx in range(idx+1, len(cls_img_pred_boxes_list)):
            forward_compute_IoU = compute_IoU(cls_img_pred_boxes_list[idx].tolist(), cls_img_pred_boxes_list[fowrard_idx].tolist())
            forward_check_overlap = check_overlap(cls_img_pred_boxes_list[idx].tolist(), cls_img_pred_boxes_list[fowrard_idx].tolist())
            if forward_compute_IoU > 0.42 or forward_check_overlap:
                reserve = False
                break
        if reserve:
            arrange_img_pred_boxes_list.append(cls_img_pred_boxes_list[idx])
            arrange_img_pred_scores_list.append(cls_img_pred_scores_list[idx])
    
    arrange_img_pred_boxes_list.append(cls_img_pred_boxes_list[-1])
    arrange_img_pred_scores_list.append(cls_img_pred_scores_list[-1])
    arrange_img_pred_boxes_list.reverse()
    arrange_img_pred_scores_list.reverse()
    return arrange_img_pred_boxes_list, arrange_img_pred_scores_list
