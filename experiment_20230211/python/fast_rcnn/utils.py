import albumentations as A
import cv2
import numpy as np

from albumentations.pytorch import ToTensorV2
from .config import DEVICE, OBJECT_AMOUNT_LIMIT
from torchvision import transforms

# this class keeps track of the training and validation loss values...
# ... and helps to get the average for each epoch as well
class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0
        
    def send(self, value):
        self.current_total += value
        self.iterations += 1
    
    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations
    
    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0

class EarlyStopping:
    def __init__(self, patient=5):
        self.patient = patient
        self.quota = patient
        self.previousLoss = 10.0

    def check(self, loss):
        if loss < self.previousLoss:
            self.previousLoss = loss
            if self.quota == self.patient:
                return 0  # better val loss
            else:
                self.quota = self.patient
                return 1  # renew quota
        else:
            self.quota -= 1
            self.previousLoss = loss
            if self.quota == 0:
                return 3  # early stop
            else:
                return 2  # quota -1

def collate_fn(batch):
    """
    To handle the data loading as different images may have different number 
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch))

def get_train_transform():
    return A.Compose([
        # A.Flip(0.5),
        # A.RandomRotate90(0.5),
        # A.MotionBlur(p=0.2),
        # A.MedianBlur(blur_limit=3, p=0.1),
        # A.Blur(blur_limit=3, p=0.1),
        A.Resize(400, 400),
        A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        A.MedianBlur(blur_limit=3, p=0.2),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=20, p=0.2),
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': 'pascal_voc',
        'label_fields': ['labels']
    })

def get_valid_transform():
    return A.Compose([
        A.Resize(400, 400),
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': 'pascal_voc', 
        'label_fields': ['labels']
    })

def show_tranformed_image(train_loader):
    """
    This function shows the transformed images from the `train_loader`.
    Helps to check whether the tranformed images along with the corresponding
    labels are correct or not.
    Only runs if `VISUALIZE_TRANSFORMED_IMAGES = True` in config.py.
    """
    if len(train_loader) > 0:
        for i in range(1):
            images, targets = next(iter(train_loader))
            images = list(image.to(DEVICE) for image in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            boxes = targets[i]['boxes'].cpu().numpy().astype(np.int32)
            sample = images[i].permute(1, 2, 0).cpu().numpy()
            for box in boxes:
                cv2.rectangle(sample,
                            (box[0], box[1]),
                            (box[2], box[3]),
                            (0, 0, 255), 2)
            cv2.imshow('Transformed image', sample)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

def find_boxes(boxes, pred_cls, target_label):
    find_boxes_list = []
    for idx, box in enumerate(boxes):
        if (pred_cls[idx] == target_label):
            find_boxes_list.append(box)
    return find_boxes_list

def arrange_object(boxes, pred_cls, scores):
    for key, value in OBJECT_AMOUNT_LIMIT.items():
        remove_amount = len(find_boxes(boxes, pred_cls, key)) - value
        if remove_amount > 0:
            for _ in range(remove_amount):
                boxes, pred_cls, scores = remove_object(boxes, pred_cls, scores, key)
    return boxes, pred_cls, scores

def remove_object(boxes, pred_cls, scores, target_label):
    occurrence = [idx for idx, cls in enumerate(pred_cls) if cls == target_label]
    del boxes[occurrence[-1]]
    del pred_cls[occurrence[-1]]
    del scores[occurrence[-1]]
    return boxes, pred_cls, scores