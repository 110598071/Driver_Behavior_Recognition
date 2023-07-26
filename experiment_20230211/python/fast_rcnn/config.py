import torch

BATCH_SIZE = 8
RESIZE_TO = 400
NUM_EPOCHS = 50
WEIGHT_DECAY = 0.005
LEARNING_RATE = 0.0005
DETECTION_THRESHOLD = 0.7
SEMI_SUPERVISED_ACCEPTANCE = 0.0
SCHEDULER_KWARGS = {
    'gamma': 0.99,
}

IOU_THRESHOLD_STEP = 0.005
IOU_THRESHOLD_UPPER = 0.95 + IOU_THRESHOLD_STEP
IOU_THRESHOLD_LOWER = 0.7

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

TRAIN_DIR = 'D:/Project/pytorch-openpose/experiment_0211/dataset/object_detection_test/train'
VALID_DIR = 'D:/Project/pytorch-openpose/experiment_0211/dataset/object_detection_test/test'
CHEAT_DIR = 'D:/Project/pytorch-openpose/experiment_0211/dataset/object_detection_test/cheat'

# classes: 0 index is reserved for background
CLASSES = [
    'background', 'hand', 'cellphone', 'steering_wheel', 'face', 'person', 'drink'
]
NUM_CLASSES = len(CLASSES)

OUT_MODEL_DIR = 'D:/Project/pytorch-openpose/experiment_0211/model/fast_rcnn'
OUT_LOSS_DIR = 'D:/Project/pytorch-openpose/experiment_0211/plot/fast_rcnn_loss'

INFERENCE_MODEL = 'D:/Project/pytorch-openpose/experiment_0211/model/fast_rcnn/model_20230520_2042.pth'

INFERENCE_DIR = 'D:/Project/pytorch-openpose/experiment_0211/dataset/object_detection_test/whole_test'
OUT_INFERENCE = 'D:/Project/pytorch-openpose/experiment_0211/dataset/object_detection_test/output'

OBJECT_AMOUNT_LIMIT = {
    'person': 1,
    'face': 1,
    'hand': 2,
    'steering_wheel': 1,
    'cellphone': 1,
    'drink': 1,
}