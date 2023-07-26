from .config import DEVICE, NUM_CLASSES, CLASSES, RESIZE_TO, DETECTION_THRESHOLD, INFERENCE_MODEL, OUT_INFERENCE, INFERENCE_DIR
from .model import create_model
from .utils import arrange_object
import numpy as np
import cv2
import os
import torch
import glob as glob
import tqdm

def inference(model, test_images, save_img, detection_threshold=DETECTION_THRESHOLD):
    if save_img:
        for cls in range(10):
            for imgPath in glob.glob(f'{OUT_INFERENCE}/c{cls}/*.jpg'):
                os.remove(imgPath)
                
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
            boxes = boxes[scores >= detection_threshold].astype(np.int32)
            pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]
            
            boxes, pred_classes, scores = arrange_object(boxes.tolist(), pred_classes[0:len(boxes)], scores[0:len(boxes)].tolist())

            boxes_list.append(boxes)
            pred_cls_list.append(pred_classes)
            scores_list.append(scores)

            if save_img:
                for j, box in enumerate(boxes):
                    cv2.rectangle(orig_image,
                                (int(box[0]), int(box[1])),
                                (int(box[2]), int(box[3])),
                                (0, 0, 255), 2)
                    cv2.putText(orig_image, pred_classes[j], 
                                (int(box[0]), int(box[3]+5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 
                                2, lineType=cv2.LINE_AA)
                cv2.imwrite(test_images[i].replace('object_detection_test/whole_test', 'object_detection_test/output'), orig_image,)
        pbar.set_description('Fast RCNN inference')
    return boxes_list, pred_cls_list, scores_list

if __name__ == '__main__':
    model = create_model(num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(INFERENCE_MODEL, map_location=DEVICE))
    model.eval()

    test_images = []
    if len(glob.glob(f'{INFERENCE_DIR}/*.jpg')) == 0:
        for subdir in os.listdir(INFERENCE_DIR):
            test_images.extend(glob.glob(f"{INFERENCE_DIR}/{subdir}/*.jpg"))
    else:
        test_images.extend(glob.glob(f"{INFERENCE_DIR}/*.jpg"))
    print(f"Test instances: {len(test_images)}")

    boxes_list, pred_cls_list, scores_list = inference(model, test_images, True)
    # print(boxes_list)
    # print(pred_cls_list)
    # print(scores_list)