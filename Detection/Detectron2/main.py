# detectron2 imports
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, LVISEvaluator
from detectron2.data import build_detection_test_loader
from detectron2.utils.visualizer import ColorMode

# other libs (you can remove unnecessary imports)

import torch, torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

from PIL import Image
import cv2
import numpy as np
import IPython
import json
import os
import json
import csv
import time
#import random
from pathlib import Path


import argparse
import os

# craete arg parser
parser = argparse.ArgumentParser(description='Detectron2 training')
parser.add_argument('--data', type=str, default='expert2', help='input path')

parser.add_argument('--weight', type=str, default='detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl', help='weight path')

parser.add_argument('--output', type=str, default='out', help='weight path')


data=parser.parse_args().data
weight_path=parser.parse_args().weight
output=parser.parse_args().output
if output=='out':
    output=data
def create_data_pairs(input_path, detectron_img_path, detectron_annot_path, dir_type = 'train'):

    img_paths = Path(input_path + 'images/'+ dir_type).glob('*.png')

    pairs = []
    for img_path in img_paths:
        file_name_tmp = str(img_path).split('/')[-1].split('.')
        file_name_tmp.pop(-1)
        file_name = '.'.join((file_name_tmp))

        label_path = Path(input_path + '/labels/' +dir_type+'/'  + file_name + '.txt')
        if label_path.is_file():

            line_img = detectron_img_path + 'images/'+ dir_type+'/' +file_name + '.png'
            line_annot = detectron_annot_path+'labels/'+ dir_type+'/' +file_name + '.txt'
            pairs.append([line_img, line_annot])

    return pairs


input_path=f'./yolo/{data}/'

train = create_data_pairs(input_path, input_path, input_path, 'train')
val = create_data_pairs(input_path, input_path, input_path, 'test')

def create_coco_format(data_pairs):
    data_list = []

    for i, path in enumerate(data_pairs):

        filename = path[0]

        img_h, img_w = cv2.imread(filename).shape[:2]

        img_item = {}
        img_item['file_name'] = filename
        img_item['image_id'] = i
        img_item['height']= img_h
        img_item['width']= img_w
        annotations = []
        with open(path[1]) as annot_file:
            lines = annot_file.readlines()
            for line in lines:
                if line[-1]=="\n":
                  box = line[:-1].split(' ')
                else:
                  box = line.split(' ')

                class_id = box[0]
                x_c = float(box[1])
                y_c = float(box[2])
                width = float(box[3])
                height = float(box[4])

                x1 = (x_c - (width/2)) * img_w
                y1 = (y_c - (height/2)) * img_h
                x2 = (x_c + (width/2)) * img_w
                y2 = (y_c + (height/2)) * img_h

                annotation = {
                    "bbox": list(map(float,[x1, y1, x2, y2])),
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": int(class_id),
                    "iscrowd": 0
                }
                annotations.append(annotation)
            img_item["annotations"] = annotations
        data_list.append(img_item)
    return data_list

train_list = create_coco_format(train)
val_list = create_coco_format(val)

for catalog_name, file_annots in [("train", train_list), ("val", val_list)]:
    DatasetCatalog.register(catalog_name, lambda file_annots = file_annots: file_annots)
    MetadataCatalog.get(catalog_name).set(thing_classes=['fault'])
metadata = MetadataCatalog.get("train")

max_iter = (int(len(train_list)/2)) * 100


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("train",)
cfg.DATALOADER.NUM_WORKERS = 1
cfg.MODEL.DEVICE = 'cuda' # cpu

#cfg.MODEL.WEIGHTS ="detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl" #  "./output/model_final.pth" #
cfg.MODEL.WEIGHTS =weight_path
cfg.SOLVER.IMS_PER_BATCH = 16
cfg.SOLVER.CHECKPOINT_PERIOD = 500
cfg.SOLVER.BASE_LR = 0.02
cfg.SOLVER.MAX_ITER = 100 # (int(len(train_list)/cfg.SOLVER.IMS_PER_BATCH )) * 10 # (train_size / batch_size) * 100
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256 # 512
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(MetadataCatalog.get("train").thing_classes)
cfg.SOLVER.STEPS = (int(cfg.SOLVER.MAX_ITER*0.95), int(cfg.SOLVER.MAX_ITER*10) )
cfg.OUTPUT_DIR = f"./output/{output}/"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

trainer = DefaultTrainer(cfg)

trainer.resume_or_load(resume=False)

import time as t
s1 = t.time()
try:
  trainer.train()
except:
  None
s2 = t.time()
print(s2-s1)


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
cfg.MODEL.DEVICE = 'cuda' # cpu
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.WEIGHTS = f"./output/{output}/model_final.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model
predictor = DefaultPredictor(cfg)

evaluator = COCOEvaluator("val", cfg, False, output_dir=f"./output/{output}/eval/")
val_loader = build_detection_test_loader(cfg, "val")
results=inference_on_dataset(trainer.model, val_loader, evaluator)
# write results on file
with open(f"./output/{output}/results.txt", 'w') as f:
    f.write(str(results['bbox']['AP50']))



