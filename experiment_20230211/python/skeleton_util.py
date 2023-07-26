import numpy as np
import math

# COMPUTE_PART = {
#     "distance": [
#         (8, 2), (14, 2), (8, 32), (14, 32), (8, 30), (14, 30), (8, 14), (28, 30)
#     ],
#     "angle": [
#         (2, 4), (4, 6), (6, 8), (2, 10), (10, 12), (12, 14)
#     ]
# }

COMPUTE_PART = {
    "distance": [],
    "angle": []
}
for i in range(0, 36, 2):
    for j in range(i+2, 36, 2):
        COMPUTE_PART['distance'].append((i, j))
        COMPUTE_PART['angle'].append((i, j))


def get_skeleton(body_estimation, oriImg):
    candidate, subset = body_estimation(oriImg)
    [c_rows, c_cols] = candidate.shape
    [s_rows, s_cols] = subset.shape

    candidate_dict = {}
    for i in range(c_rows):
        candidate_dict[candidate[i, 3]] = [candidate[i, 0], candidate[i, 1]]

    skeletons_score_list = []
    for i in range(s_rows):
        skeletons_score_list.append(subset[i, s_cols - 2])

    max_score_s_row = np.argmax(skeletons_score_list)
    rowList = []
    for j in range(s_cols):
        if j < s_cols - 2:
            if subset[max_score_s_row, j] < 0:
                rowList.append(-1)
                rowList.append(-1)
            else:
                rowList.append(float(candidate_dict[subset[max_score_s_row, j]][0]))
                rowList.append(float(candidate_dict[subset[max_score_s_row, j]][1]))
    return rowList


def compute_feature(rowList):
    action_data = []

    action_data.append(1 if rowList[32] != '-1' else 0)
    action_data.append(1 if rowList[34] != '-1' else 0)

    for key, value in COMPUTE_PART.items():
        for part in value:
            action_data.append(check_and_compute_parts(part[0], part[1], rowList, key))

    return action_data


def check_and_compute_parts(p1, p2, rowList, key):
    if rowList[p1] != '-1' and rowList[p2] != '-1':
        if key == "distance":
            return compute_distance(rowList[p1], rowList[p1 + 1], rowList[p2], rowList[p2 + 1])
        elif key == "angle":
            return compute_angle(rowList[p1], rowList[p1 + 1], rowList[p2], rowList[p2 + 1])
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
    print(len(COMPUTE_PART['angle']))
    print(COMPUTE_PART['angle'])
