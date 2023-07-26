import csv
import math
import numpy as np

TRAIN_CSV = '../csv/train_skeleton_output.csv'
TEST_CSV = '../csv/test_skeleton_output.csv'

COMPUTE_DISTANCE_PART = [(10, 4), (16, 4), (10, 34), (16, 34), (10, 32), (16, 32), (10, 16), (30, 32)]
COMPUTE_ANGLE_PART = [(4, 6), (6, 8), (8, 10), (4, 12), (12, 14), (14, 16)]

def compute_feature(csv_path):
    action_data = []
    action_label = []

    f = open(csv_path,'r',encoding = 'utf8')
    plots = csv.reader(f, delimiter=',')

    for i, item in enumerate(plots):
        if i != 0:
            row = []
            if item[36] != '-1':
                row.append(1)
            else:
                row.append(0)

            for parts in COMPUTE_DISTANCE_PART:
                row.append(check_and_compute_distance(parts[0], parts[1], item))
            for parts in COMPUTE_ANGLE_PART:
                row.append(check_and_compute_angle(parts[0], parts[1], item))
            action_data.append(row)
            action_label.append(int(item[1]))
    f.close()
    return action_data, action_label

def check_and_compute_distance(p1, p2, item):
    if item[p1] != '-1' and item[p2] != '-1':
        return compute_distance(float(item[p1]), float(item[p1+1]), float(item[p2]), float(item[p2+1]))
    else:
        return -1

def check_and_compute_angle(p1, p2, item):
    if item[p1] != '-1' and item[p2] != '-1':
        return compute_angle(float(item[p1]), float(item[p1+1]), float(item[p2]), float(item[p2+1]))
    else:
        return -1

def compute_distance(x1, y1, x2, y2):
    a = x2 - x1
    b = y2 - y1
    return np.round(math.sqrt(math.pow(a, 2) + math.pow(b, 2)), 3)

def compute_angle(x1, y1, x2, y2):
    a = x2 - x1
    b = y2 - y1
    if a == 0:
        if b > 0: return 90
        else: return 270
    else:
        angle = np.round(math.degrees(math.atan(np.round(b/a, 3))), 3)
        if a < 0:
            return 180 + angle
        elif a > 0 and b < 0:
            return 360 + angle
        else:
            return angle     

def get_train_and_test_data():
    data_train, label_train = compute_feature(TRAIN_CSV)
    data_test, label_test = compute_feature(TEST_CSV)
    return (np.asarray(data_train), np.asarray(label_train)), (np.asarray(data_test), np.asarray(label_test))

if __name__ == '__main__':
    action_data, action_label = compute_feature(TRAIN_CSV)
    action_data.reverse()
    for i in action_data:
        print(i)
    # print(action_label)