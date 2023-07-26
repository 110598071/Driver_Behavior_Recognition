from .config import NUM_CLASSES, DEVICE, INFERENCE_MODEL, RESIZE_TO, OBJECT_AMOUNT_LIMIT
from .model import create_model
from .inference import inference
import torch
import numpy as np
import math
from tqdm import tqdm
from collections import namedtuple
Rectangle = namedtuple('Rectangle', 'x_min, y_min, x_max, y_max')
Point = namedtuple('Point', 'x, y')

APPERENCE_OBJECT = ['cellphone', 'drink']
COMPUTE_DISTANCE_OBJECT = [
    ['right_hand', 'person'], ['left_hand', 'person'],
    ['right_hand', 'face'], ['left_hand', 'face'],
    ['right_hand', 'cellphone'], ['left_hand', 'cellphone'],
    ['right_hand', 'drink'], ['left_hand', 'drink'],
    ['right_hand', 'left_hand'],
    ['face', 'cellphone'],
    ['face', 'drink'],
]
COMPUTE_OVERLAP_OBJECT = [
    ['right_hand', 'cellphone'], ['left_hand', 'cellphone'],
    ['right_hand', 'steering_wheel'], ['left_hand', 'steering_wheel'],
    ['right_hand', 'face'], ['left_hand', 'face'],
    ['right_hand', 'drink'], ['left_hand', 'drink'],
    ['face', 'cellphone'],
    ['face', 'drink'],
]

class ImageObjectDetection:
    def __init__(self, boxes, pred_cls, scores, keypoints_rowList):
        self.boxes = boxes
        self.pred_cls = pred_cls
        self.scores = scores
        self.keypoints_rowList = keypoints_rowList
        self.arrange_object()
        self.find_right_left_hand()
        self.face_direction = self.judge_face_direction()

        self.hand_complete = self.complete_right_left_hand()
        self.eye_and_ear_both_appear = self.find_eye_both_appear()
        self.hand_on_the_same_side = self.judge_hand_on_the_same_side()
        self.face_area = self.compute_face_area()
        self.upper_arm_angle = self.compute_upper_arm_angle()
        self.ear_to_eye_angle = self.compute_ear_to_eye_angle()
        self.distance_between_eye_and_block_ear = self.compute_distance_between_eye_and_block_ear()
        self.vervical_distance_between_cellphone_and_face = self.compute_vervical_distance_between_cellphone_and_face()

        self.appearance_features = self.get_appearance_features()
        self.distance_features = self.get_distance_features()
        self.overlap_features = self.get_overlap_features()

        self.final_features = []
        self.final_features.extend(self.hand_complete) #2
        self.final_features.extend(self.eye_and_ear_both_appear) #2
        self.final_features.append(self.hand_on_the_same_side) #1
        self.final_features.append(self.face_area) #1
        self.final_features.append(self.upper_arm_angle) #1
        self.final_features.append(self.ear_to_eye_angle) #1
        self.final_features.append(self.distance_between_eye_and_block_ear) #1
        self.final_features.append(self.vervical_distance_between_cellphone_and_face) #1
        self.final_features.extend(self.appearance_features) #2
        self.final_features.extend(self.distance_features) #11
        self.final_features.extend(self.overlap_features) #10
        self.final_features = [float(i) for i in self.final_features] 

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

    def find_right_left_hand(self):
        # openpose right/left wrist keypoint
        r_wrist_point = Point._make([int(self.keypoints_rowList[8]), int(self.keypoints_rowList[9])])
        l_wrist_point = Point._make([int(self.keypoints_rowList[14]), int(self.keypoints_rowList[15])])

        hand_boxes = self.find_boxes('hand')
        record_side_list = []
        record_dst_list = []
        for box in hand_boxes:
            # dst_list represent distance between hand_boxes and openpose wrist keypoints
            dst_list = []
            for wrist_point in [r_wrist_point, l_wrist_point]:
                dst_list.append(compute_distance(wrist_point, Point._make(get_box_center_coor(Rectangle._make(box)))))

            # identify current hand_box is more closer to which side of wrist keypoint
            min_idx = np.argmin(dst_list)
            # if the distance between current hand_box and wrist keypoint is too far, then the assigned side should be opposite
            # ex. (right)hand_box / wrist keypoint only detect left side, which mean current box will be assigned to left hand because distance to left wrist keypoint is more shorter
            if (dst_list[min_idx] > 60):
                min_idx = (0 if min_idx == 1 else 1)

            # record current hand_box is more likely to be which side hand and the distance to wrist keypoint
            record_side_list.append('right_hand' if min_idx == 0 else 'left_hand')
            record_dst_list.append(dst_list[min_idx])

        # if we have two(max) hand_box and both of it is assigned to the same side, then compare their distance to wrist keypoint
        # the shorter can keep it's hand side, the longer should change it's hand side
        if len(record_side_list) == 2 and record_side_list[0] == record_side_list[1]:
            record_side_list[np.argmax(record_dst_list)] = ('right_hand' if record_side_list[np.argmax(record_dst_list)] == 'left_hand' else 'left_hand')

        # replace 'hand' label in pred_cls to 'right_hand'/'left_hand'
        hand_count = 0
        new_pred_cls = []
        for idx, box in enumerate(self.boxes):
            if (self.pred_cls[idx] == 'hand'):
                new_pred_cls.append(record_side_list[hand_count])
                hand_count += 1
            else:
                new_pred_cls.append(self.pred_cls[idx])
        self.pred_cls = new_pred_cls

    def judge_face_direction(self):
        person_boxes = self.find_boxes('person')
        wheel_boxes = self.find_boxes('steering_wheel')

        if len(person_boxes) != 0 and len(wheel_boxes) != 0:
            person_center_point = Point._make(get_box_center_coor(Rectangle._make(person_boxes[0])))
            wheel_center_point = Point._make(get_box_center_coor(Rectangle._make(wheel_boxes[0])))
            if person_center_point.x < wheel_center_point.x:
                return 'right'
            else:
                return 'left'
        else:
            return 'not_found'

    def complete_right_left_hand(self):
        r_hand_complete = 0
        l_hand_complete = 0

        r_hand_boxes = self.find_boxes('right_hand')
        l_hand_boxes = self.find_boxes('left_hand')

        r_wrist_origin_point = Point._make([int(self.keypoints_rowList[8]), int(self.keypoints_rowList[9])])
        l_wrist_origin_point = Point._make([int(self.keypoints_rowList[14]), int(self.keypoints_rowList[15])])
        r_wrist_point = Point._make([int(self.keypoints_rowList[8]), int(self.keypoints_rowList[9])])
        l_wrist_point = Point._make([int(self.keypoints_rowList[14]), int(self.keypoints_rowList[15])])
        r_elbow_point = Point._make([int(self.keypoints_rowList[6]), int(self.keypoints_rowList[7])])
        l_elbow_point = Point._make([int(self.keypoints_rowList[12]), int(self.keypoints_rowList[13])])

        if len(r_hand_boxes) == 0 and r_wrist_origin_point.x != -1 and r_wrist_origin_point.y != -1:
            r_hand_corner_point = Point._make([int(r_wrist_point.x + (r_wrist_point.x - r_elbow_point.x) * 0.6), int(r_wrist_point.y + (r_wrist_point.y - r_elbow_point.y) * 0.6)])
            x_max = np.max([r_hand_corner_point.x, r_wrist_point.x])
            x_min = np.min([r_hand_corner_point.x, r_wrist_point.x])
            y_max = np.max([r_hand_corner_point.y, r_wrist_point.y])
            y_min = np.min([r_hand_corner_point.y, r_wrist_point.y])

            self.boxes.append([x_min, y_min, x_max, y_max])
            self.pred_cls.append('right_hand')
            self.scores.append(0.96)
            r_hand_complete = 1
        if len(l_hand_boxes) == 0 and l_wrist_origin_point.x != -1 and l_wrist_origin_point.y != -1:
            l_hand_corner_point = Point._make([int(l_wrist_point.x + (l_wrist_point.x - l_elbow_point.x) * 0.6), int(l_wrist_point.y + (l_wrist_point.y - l_elbow_point.y) * 0.6)])
            x_max = np.max([l_hand_corner_point.x, l_wrist_point.x])
            x_min = np.min([l_hand_corner_point.x, l_wrist_point.x])
            y_max = np.max([l_hand_corner_point.y, l_wrist_point.y])
            y_min = np.min([l_hand_corner_point.y, l_wrist_point.y])

            self.boxes.append([x_min, y_min, x_max, y_max])
            self.pred_cls.append('left_hand')
            self.scores.append(0.96)
            l_hand_complete = 1
        return [r_hand_complete, l_hand_complete]
    
    def judge_hand_on_the_same_side(self):
        r_hand_boxes = self.find_boxes('right_hand')
        l_hand_boxes = self.find_boxes('left_hand')
        face_boxes = self.find_boxes('face')
        r_elbow_point = Point._make([int(self.keypoints_rowList[6]), int(self.keypoints_rowList[7])])
        l_elbow_point = Point._make([int(self.keypoints_rowList[12]), int(self.keypoints_rowList[13])])

        if len(face_boxes) != 0:
            face_point = Point._make(get_box_center_coor(Rectangle._make(face_boxes[0])))
            if len(r_hand_boxes) != 0:
                r_point = Point._make(get_box_center_coor(Rectangle._make(r_hand_boxes[0])))
            elif r_elbow_point.x != -1:
                r_point = r_elbow_point
            else:
                return 1 # hand not found
            if len(l_hand_boxes) != 0:
                l_point = Point._make(get_box_center_coor(Rectangle._make(l_hand_boxes[0])))
            elif l_elbow_point.x != -1:
                l_point = l_elbow_point
            else:
                return 1 # hand not found
            if (face_point.x > r_point.x and face_point.x < l_point.x) or (face_point.x < r_point.x and face_point.x > l_point.x):
                return 2 # different side
            else:
                return 3 # same side
        else:
            return 0 # face not found
        
    def find_eye_both_appear(self):
        r_eye_point = Point._make([int(self.keypoints_rowList[28]), int(self.keypoints_rowList[29])])
        l_eye_point = Point._make([int(self.keypoints_rowList[30]), int(self.keypoints_rowList[31])])
        r_ear_point = Point._make([int(self.keypoints_rowList[32]), int(self.keypoints_rowList[33])])
        l_ear_point = Point._make([int(self.keypoints_rowList[34]), int(self.keypoints_rowList[35])])

        if r_eye_point.x != -1 and l_eye_point.x != -1:
            eye_both_appear = 1
        else:
            eye_both_appear = 0
        if r_ear_point.x != -1 and l_ear_point.x != -1:
            ear_both_appear = 1
        else:
            ear_both_appear = 0
        return [eye_both_appear, ear_both_appear]
        
    def compute_face_area(self):
        face_boxes = self.find_boxes('face')
        if len(face_boxes) != 0:
            return compute_box_area(Rectangle._make(face_boxes[0]))
        else:
            return 0.0 # face not found
        
    def compute_upper_arm_angle(self):
        if self.face_direction == 'not_found':
            return 0
        else:
            if self.face_direction == 'right':
                shoulder_point = Point._make([int(self.keypoints_rowList[4]), 400 - int(self.keypoints_rowList[5])])
                elbow_point = Point._make([int(self.keypoints_rowList[6]), 400 - int(self.keypoints_rowList[7])])
            else:
                shoulder_point = Point._make([int(self.keypoints_rowList[10]), 400 - int(self.keypoints_rowList[11])])
                elbow_point = Point._make([shoulder_point.x * 2 - int(self.keypoints_rowList[12]), 400 - int(self.keypoints_rowList[13])])
            if shoulder_point.x != -1 and elbow_point.x != -1:
                return compute_angle(shoulder_point, elbow_point)
            else:
                return 0
            
    def compute_ear_to_eye_angle(self):
        if self.face_direction == 'not_found':
            return 0
        else:
            if self.face_direction == 'right':
                eye_point = Point._make([int(self.keypoints_rowList[28]),  400 - int(self.keypoints_rowList[29])])
                ear_point = Point._make([int(self.keypoints_rowList[32]),  400 - int(self.keypoints_rowList[33])])
            else:
                eye_point = Point._make([int(self.keypoints_rowList[30]),  400 - int(self.keypoints_rowList[31])])
                ear_point = Point._make([eye_point.x * 2 - int(self.keypoints_rowList[34]),  400 - int(self.keypoints_rowList[35])])
            if eye_point.x != -1 and ear_point.x != -1:
                return compute_angle(ear_point, eye_point)
            else:
                return 0
            
    def compute_distance_between_eye_and_block_ear(self):
        if self.face_direction == 'not_found':
            return 0
        else:
            if self.face_direction == 'right':
                eye_point = Point._make([int(self.keypoints_rowList[30]), int(self.keypoints_rowList[31])])
                ear_point = Point._make([int(self.keypoints_rowList[34]), int(self.keypoints_rowList[35])])
            else:
                eye_point = Point._make([int(self.keypoints_rowList[28]), int(self.keypoints_rowList[29])])
                ear_point = Point._make([int(self.keypoints_rowList[32]), int(self.keypoints_rowList[33])])
            if eye_point.x != -1 and ear_point.x != -1:
                return compute_distance(eye_point, ear_point)
            else:
                return 0
        
    def compute_vervical_distance_between_cellphone_and_face(self):
        cellphone_boxes = self.find_boxes('cellphone')
        face_boxes = self.find_boxes('face')
        if len(cellphone_boxes) != 0 and len(face_boxes) != 0:
            cellphone_vertical_val = get_box_center_coor(Rectangle._make(cellphone_boxes[0]))[1]
            face_vertical_val = get_box_center_coor(Rectangle._make(face_boxes[0]))[1]
            return cellphone_vertical_val - face_vertical_val
        else:
            return 0

    def remove_object(self, target_label):
        occurrence = [idx for idx, cls in enumerate(self.pred_cls) if cls == target_label]
        del self.boxes[occurrence[-1]]
        del self.pred_cls[occurrence[-1]]
        del self.scores[occurrence[-1]]

    def get_appearance_features(self):
        appearance_features = []
        for obj in APPERENCE_OBJECT:
            appearance_features.append(len(self.find_boxes(obj)))
        return appearance_features
    
    def get_distance_features(self):
        distance_features = []
        for target1, target2 in COMPUTE_DISTANCE_OBJECT:
            distance_features.append(self.compute_single_distance_feature(target1, target2))
        return distance_features
    
    def get_overlap_features(self):
        overlap_features = []
        for target1, target2 in COMPUTE_OVERLAP_OBJECT:
            overlap_features.append(self.compute_single_overlap_feature(target1, target2))
        return overlap_features

    def compute_single_distance_feature(self, target1, target2):
        result_dist = 0
        for target1_box in self.find_boxes(target1):
            target1_center_point = Point._make(get_box_center_coor(Rectangle._make(target1_box)))

            for target2_box in self.find_boxes(target2):
                target2_center_point = Point._make(get_box_center_coor(Rectangle._make(target2_box)))
                result_dist += compute_distance(target1_center_point, target2_center_point)
        return result_dist

    def compute_single_overlap_feature(self, target1, target2):
        result_area = 0
        for target1_box in self.find_boxes(target1):
            target1_rectangle = Rectangle._make(target1_box)

            for target2_box in self.find_boxes(target2):
                target2_rectangle = Rectangle._make(target2_box)
                result_area += compute_overlap_area(target1_rectangle, target2_rectangle)
        return result_area
    
    def get_final_features(self):
        return self.final_features

    def print_object(self):
        print('boxes: ', self.boxes)
        print('pred_cls: ', self.pred_cls)
        print('scores: ', self.scores)
        print()
        print('hand_complete: ', self.hand_complete)
        print('eye_and_ear_both_appear: ', self.eye_and_ear_both_appear)
        print('hand_on_the_same_side: ', self.hand_on_the_same_side)
        print('face_area: ', self.face_area)
        print('upper_arm_angle: ', self.upper_arm_angle)
        print('ear_to_eye_angle: ', self.ear_to_eye_angle)
        print('distance_between_eye_and_block_ear: ', self.distance_between_eye_and_block_ear)
        print('vervical_distance_between_cellphone_and_face: ', self.vervical_distance_between_cellphone_and_face)
        print('appearance_features: ', self.appearance_features)
        print('distance_features: ', self.distance_features)
        print('overlap_features: ', self.overlap_features)
        print('final_features: ', self.final_features)

def feature_computation(imgPathList, keypoints_list, label_list):
    model = create_model(num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(INFERENCE_MODEL, map_location=DEVICE))
    model.eval()

    final_features = []
    boxes_list, pred_cls_list, scores_list = inference(model, imgPathList, False)
    for idx, boxes in enumerate(boxes_list):
        if imgPathList[idx].split('\\')[-1] in keypoints_list[label_list[idx]]:
            keypoint_row_list = keypoints_list[label_list[idx]][imgPathList[idx].split('\\')[-1]]
        else:
            keypoint_row_list = [-1 for _ in range(38)]
        image_object_detection = ImageObjectDetection(boxes, pred_cls_list[idx], scores_list[idx], keypoint_row_list)
        final_features.append(image_object_detection.get_final_features())
        # image_object_detection.print_object()
    final_features = normalize_final_features(final_features)
    return final_features

def normalize_final_features(final_features):
    features_list = [[] for _ in range(len(final_features[0]))] # store independent featue data (origin is independent image)
    for img_feature in final_features:
        for feature_idx, feature in enumerate(img_feature):
            features_list[feature_idx].append(feature)

    feature_min_list = []
    feature_max_list = []
    for feature in features_list:
        feature_min_list.append(np.min(feature))
        feature_max_list.append(np.max(feature))

    processed_final_features = [[] for _ in range(len(final_features))]
    pbar = tqdm(final_features)
    for img_idx, img_feature in enumerate(pbar):
        for feature_idx, feature in enumerate(img_feature):
            feature_min = feature_min_list[feature_idx]
            feature_max = feature_max_list[feature_idx]
            if int(feature_max - feature_min) == 0:
                processed_feature = feature
            else:
                processed_feature = np.round((feature - feature_min) / (feature_max - feature_min), 4)
            processed_final_features[img_idx].append(processed_feature)
        pbar.set_description('normalize features:')
    return processed_final_features

def get_box_center_coor(box):
    return ((np.floor([(box.x_max + box.x_min)/2, (box.y_max + box.y_min)/2])).astype(int)).tolist()

def compute_box_area(box):
    return (box.x_max - box.x_min)*(box.y_max - box.y_min)

def compute_overlap_area(box1, box2):
    dx = min(box1.x_max, box2.x_max) - max(box1.x_min, box2.x_min)
    dy = min(box1.y_max, box2.y_max) - max(box1.y_min, box2.y_min)
    if (dx>=0) and (dy>=0):
        return dx*dy
    else:
        return 0
    
def compute_distance(point1, point2):
    if point1.x == -1 or point2.x == -1:
        return 1000
    else:
        return int(np.floor(math.sqrt(math.pow(point1.x - point2.x, 2) + math.pow(point1.y - point2.y, 2))))
    
def compute_angle(point1, point2):
    a = point2.x - point1.x
    b = point2.y - point1.y
    if a == 0:
        if b > 0:
            return 90
        else:
            return 270
    else:
        angle = np.round(math.degrees(math.atan(np.round(b / a, 3))), 3)
        if a < 0:
            return 180 + angle
        elif a > 0 and b < 0:
            return 360 + angle
        else:
            return angle

if __name__ == '__main__':
    imgPathList = ['D:/Project/pytorch-openpose/experiment_0211/dataset/object_detection_test/test/c0/011.jpg']
    imgPathList.append('D:/Project/pytorch-openpose/experiment_0211/dataset/object_detection_test/test/c0/020.jpg')
    imgPathList.append('D:/Project/pytorch-openpose/experiment_0211/dataset/object_detection_test/test/c2/041.jpg')
    final_features = feature_computation(imgPathList)
    print(final_features)