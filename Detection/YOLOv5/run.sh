#!/bin/bash
python train.py --img 640 --epochs 100   --batch-size 16 --cfg yolov5l.yaml --weights yolov5l.pt --name expert --data expert.yaml

python val.py --img 640   --batch-size 16 --weights ./runs/train/expert/weights/best.pt  --data expert.yaml

python segment/train.py --weights yolov5l-seg.pt  --epochs 100 --img 640 --data expert.yaml --name expert

python segment/val.py --weights ./runs/train-seg/expert/weights/best.pt  --img 640 --data expert.yaml --batch-size 16















