from pytorch_config import RCNN_DATA_DIR, CNN_MODEL_PATH, get_keypoints_feautres_list
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import fast_rcnn.feature_computation
import glob
import pytorch_model
from datetime import datetime
import pytorch_util
import numpy as np

class ObjectPatternDetection:
    def __init__(self, DATA_DIR, CNN_MODEL_PATH):
        self.final_label_list = []
        self.label_test = []
        self.test_imgPathList = []
        keypoints_test_list = get_keypoints_feautres_list('test')

        for cls in range(10):
            self.label_test.extend([cls for _ in range(len(glob.glob(f"{DATA_DIR}/test/c{cls}/*.jpg")))])
            self.test_imgPathList.extend(glob.glob(f"{DATA_DIR}/test/c{cls}/*.jpg"))

        model = pytorch_model.get_pretrained_ResNet("101")
        self.test_output_list = pytorch_util.test_image(model, CNN_MODEL_PATH, self.test_imgPathList)
        self.test_object_feature_list = fast_rcnn.feature_computation.feature_computation(self.test_imgPathList, keypoints_test_list, self.label_test)

    def compute_final_output(self):
        for img_idx, output in enumerate(self.test_output_list):
            final_output = []
            for idx, score in enumerate(output):
                final_output.append(self.find_pattern(idx, img_idx) + score)
            self.final_label_list.append(np.argmax(final_output))

    def plot_confusion_matrix(self):
        cf_matrix = confusion_matrix(self.label_test, self.final_label_list)

        ticks = [list(range(10))]
        df = pd.DataFrame(cf_matrix, columns=ticks, index=ticks)

        plt.figure()
        sns.heatmap(df, fmt='', annot=True)
        plt.title('confusion_matrix')
        plt.ylabel('actual label')
        plt.xlabel('predicted label')
        plt.savefig('../plot/Confusion/feature_voting/Confusion' + datetime.now().strftime("_%Y%m%d_%H%M") + ".jpg", dpi=300)

    def find_pattern(self, idx, img_idx):
        point = 0
        object_feature = self.test_object_feature_list[img_idx]
        if (idx == 0):
            if object_feature[2] == 0 and object_feature[11] < 9 and object_feature[12] < 9: point += 1
            if object_feature[20] + object_feature[21] > 3300: point += 0.3
        elif (idx == 1):
            if object_feature[2] == 1 and object_feature[18] + object_feature[19] > 500: point += 0.3
            if object_feature[18] > 500: point += 0.3
        elif (idx == 2):
            if object_feature[2] == 1 and object_feature[18] + object_feature[19] > 500: point += 0.2
            if object_feature[18] > 500: point += 0.2
            if object_feature[22] > 400: point += 0.3
            if object_feature[26] > 300: point += 1.5
        elif (idx == 3):
            if object_feature[2] == 1 and object_feature[18] + object_feature[19] > 500: point += 0.2
            if object_feature[19] > 300: point += 1.5
        elif (idx == 4):
            if object_feature[2] == 1 and object_feature[23] > 100: point += 1
            if object_feature[26] > 100: point += 0.3
        elif (idx == 5):
            if object_feature[7] > 80 and object_feature[10] > 130 and object_feature[15] > 120: point += 1.5
        elif (idx == 6):
            if object_feature[6] == 1: point += 1
            if object_feature[24] > 0: point += 2
            if object_feature[25] > 0: point += 2
        elif (idx == 7):
            if object_feature[0] == 0: point += 0.2
            if object_feature[7] < 6 and object_feature[9] < 4 and object_feature[11] < 6 and object_feature[15] < 6: point += 0.8
            if object_feature[20] < 100: point += 0.3
        elif (idx == 8):
            if object_feature[22] + object_feature[23] > 200 and object_feature[9] < 85: point += 1.5
        elif (idx == 9):
            if object_feature[20] + object_feature[21] > 2400: point += 0.5
        return point
    
if __name__ == '__main__':
    objectPatternDetection = ObjectPatternDetection(RCNN_DATA_DIR, CNN_MODEL_PATH)
    objectPatternDetection.compute_final_output()
    objectPatternDetection.plot_confusion_matrix()