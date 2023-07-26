import torch
import cv2
import numpy as np
import os
import glob as glob
import random

from xml.etree import ElementTree as et
from .config import CLASSES, RESIZE_TO, TRAIN_DIR, VALID_DIR, BATCH_SIZE, CHEAT_DIR
from torch.utils.data import Dataset, DataLoader
from .utils import collate_fn, get_train_transform, get_valid_transform, show_tranformed_image

class ObjectDataset(Dataset):
    def __init__(self, dir_path_list, width, height, classes, transforms=None):
        self.transforms = transforms
        self.dir_path_list = dir_path_list
        self.height = height
        self.width = width
        self.classes = classes
        
        self.image_paths = []
        for dir_path in self.dir_path_list:
            for subdir in os.listdir(dir_path):
                self.image_paths.extend(glob.glob(f"{dir_path}/{subdir}/*.jpg"))
        self.all_images = [image_path.split('\\')[-1] for image_path in self.image_paths]
        self.all_images = sorted(self.all_images)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_resized = cv2.resize(image, (self.width, self.height))
        image_resized /= 255.0
        # image_resized = cv2.cvtColor(image_resized, cv2.COLOR_RGB2GRAY)
        # image_resized = np.stack((image_resized,)*3, axis=-1)
        
        annot_file_path = image_path.replace('.jpg', '.xml')
        
        boxes = []
        labels = []
        tree = et.parse(annot_file_path)
        root = tree.getroot()
        
        image_width = image.shape[1]
        image_height = image.shape[0]
        
        
        for member in root.findall('object'):
            labels.append(self.classes.index(member.find('name').text))
            
            # xmin = left corner x-coordinates
            xmin = int(member.find('bndbox').find('xmin').text)
            # xmax = right corner x-coordinates
            xmax = int(member.find('bndbox').find('xmax').text)
            # ymin = left corner y-coordinates
            ymin = int(member.find('bndbox').find('ymin').text)
            # ymax = right corner y-coordinates
            ymax = int(member.find('bndbox').find('ymax').text)
            
            # resize the bounding boxes according to the...
            # ... desired `width`, `height`
            xmin_final = (xmin/image_width)*self.width
            xmax_final = (xmax/image_width)*self.width
            ymin_final = (ymin/image_height)*self.height
            yamx_final = (ymax/image_height)*self.height
            
            boxes.append([xmin_final, ymin_final, xmax_final, yamx_final])
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        image_id = torch.tensor([idx])
        target["image_id"] = image_id

        if self.transforms:
            sample = self.transforms(image = image_resized,
                                     bboxes = target['boxes'],
                                     labels = labels)
            image_resized = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])
            
        return image_resized, target

    def __len__(self):
        return len(self.image_paths)
    
class InferenceDataset(Dataset):
    def __init__(self, imgPath_list, boxes_list, pred_cls_list, width, height, classes, transforms=None):
        self.imgPath_list = imgPath_list
        self.boxes_list = boxes_list
        self.height = height
        self.width = width
        self.transforms = transforms

        self.labels = []
        for pred_cls in pred_cls_list:
            label = []
            for cls in pred_cls:
                label.append(classes.index(cls))
            self.labels.append(label)
    
    def __getitem__(self, idx):
        image_path = self.imgPath_list[idx]
        image = cv2.imread(image_path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_resized = cv2.resize(image, (self.width, self.height))
        image_resized /= 255.0

        boxes = self.boxes_list[idx]
        labels = self.labels[idx]

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        image_id = torch.tensor([idx])
        target["image_id"] = image_id

        if self.transforms:
            sample = self.transforms(image = image_resized,
                                     bboxes = target['boxes'],
                                     labels = labels)
            image_resized = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])
            
        return image_resized, target

    def __len__(self):
        return len(self.imgPath_list)

# train_dataset = ObjectDataset([TRAIN_DIR, CHEAT_DIR], RESIZE_TO, RESIZE_TO, CLASSES, get_train_transform())
train_dataset = ObjectDataset([TRAIN_DIR], RESIZE_TO, RESIZE_TO, CLASSES, get_train_transform())
valid_dataset = ObjectDataset([VALID_DIR], RESIZE_TO, RESIZE_TO, CLASSES, get_valid_transform())
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    collate_fn=collate_fn
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    collate_fn=collate_fn
)

if __name__ == '__main__':
    dataset = ObjectDataset(
        TRAIN_DIR, RESIZE_TO, RESIZE_TO, CLASSES
    )
    print(f"Number of training images: {len(dataset)}")
    
    def visualize_sample(image, target):
        for i in range(len(target['boxes'])):
            box = target['boxes'][i]
            label = CLASSES[target['labels'][i]]
            cv2.rectangle(
                image, 
                (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                (0, 255, 0), 1
            )
            cv2.putText(
                image, label, (int(box[0]), int(box[1]-5)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1
            )
        cv2.imshow('Image', image)
        cv2.waitKey(0)
        
    NUM_SAMPLES_TO_VISUALIZE = 5
    for _ in range(NUM_SAMPLES_TO_VISUALIZE):
        image, target = dataset[random.randint(0, len(dataset)-1)]
        visualize_sample(image, target)