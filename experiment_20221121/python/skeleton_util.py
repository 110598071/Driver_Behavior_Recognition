import numpy as np

import sys
sys.path.append("../..")
from src.body import Body

def get_skeleton(body_estimation, oriImg, class_img_name, data_class_name):
    candidate, subset = body_estimation(oriImg)
    [c_rows, c_cols] = candidate.shape
    [s_rows, s_cols] = subset.shape

    if len(candidate) == 0:
        return

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
        # if subset[i, s_cols-2] < TOTAL_SCORE_THRESHOLD or subset[i, s_cols-1] < TOTAL_PARTS_THRESHOLD :
        #     break
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