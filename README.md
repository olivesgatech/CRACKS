# CRACKS

## Abstract
Crowdsourcing annotations has created a paradigm shift in the availability of labeled data for machine learning. Availability of large datasets has accelerated progress in common knowledge applications involving visual and language data. However, specialized applications that require expert labels lag in data availability. One such application is fault segmentation in subsurface imaging. Detecting, tracking, and analyzing faults has broad societal implications in predicting fluid flows, earthquakes, and storing excess atmospheric CO2. However, delineating faults with current practices is a labor-intensive activity that requires precise analysis of subsurface imaging data by geophysicists. In this paper, we propose the CRACKS dataset to detect and segment faults in subsurface images by utilizing crowdsourced resources. We leverage Amazon Mechanical Turk to obtain fault delineations from sections of the Netherlands North Sea subsurface images from (i) 26 novices who have no exposure to subsurface data and were shown a video describing and labeling faults, (ii) 8 practitioners who have previously interacted and worked on subsurface data, (iii) one geophysicist to label 7636 faults in the region. Note that all novices, practitioners, and the expert segment faults on the same subsurface volume with disagreements between and among the novices and practitioners. Additionally, each fault annotation is equipped with the confidence level of the annotator. The paper provides benchmarks on detecting and segmenting the expert labels, given the novice and practitioner labels. Additional details along with the dataset links and codes are available at https://alregib.ece.gatech.edu/cracks-crowdsourcing-resources-for-analysis-and-categorization-of-key-subsurface-faults/.



## Dataset

**Images and Labels**: https://zenodo.org/records/11559387

**Images** The images for the full 400 images corresponding to each seismic section are placed in a folder called images.zip. Every image is named with a convention that denotes the position within the 3D seismic volume that the section is drawn from.

**Labels** The labels exist within a separate folder called Fault Segmentations.zip. This folder contains 35 directories that each correspond to the annotations associated with our 26 novices, 8 practitioners, and the single expert. The directories are named in an intuitive manner to indicate which annotator created the associated labels. Examples include novice01, practitioner2, and expert. Within each folder named by the associated annotator are the fault annotations for each seismic section that that specific annotator worked on. The label files exist in a .png format with a naming convention that indicates which seismic section from the overall volume that these labels correspond to. As discussed in the main paper, every label file has three colors that indicates confident existence of fault (blue), uncertain existence of fault (green), and confidence of the non-existence of a fault. 

## Code Usage
**YOLOv5 Detection Experiments**

1. Go to the **Detection/YOLOv5** directory
2. Train the model with expert labels:
```
python train.py --img 640 --epochs 100   --batch-size 16 --cfg yolov5l.yaml --weights yolov5l.pt --name expert --data expert.yaml
```
3. Evaluate the trained model with 
```
python val.py --img 640   --batch-size 16 --weights ./runs/train/expert/weights/best.pt  --data expert.yaml
```
**YOLOv5 Instance Segmentation Experiments**

1. Go to the **Detection/YOLOv5** directory
2. Train the model with expert labels:
```
python segment/train.py --weights yolov5l-seg.pt  --epochs 100 --img 640 --data expert.yaml --name expert
```
3. Evaluate the trained model with 
```
python segment/val.py --weights ./runs/train-seg/expert/weights/best.pt  --img 640 --data expert.yaml --batch-size 16
```

**FastRCNN, RetinaNet, DETR Detection Experiments**

1. Go to the **Detection/Detectron2** directory
2. Install the Detectron 2 livrary following installation guidelines in INSTALL.md
3. Train the model using main.py 
```
python main.py --data expert --output expert 
```




**Self-Supervised Experiments**

1. Go to the **ssl** directory
2. Train the backbone network with SimCLR loss function and object crops using main_supcon.py\
a) An example of a script would be: \
python main_supcon.py --dataset 'Fault' --model 'resnet18' --batch_size 64
3. Attach the segmentation head of choice and freeze the backbone network with the weights from the previous step \
a) Set the annotator folder that you want to finetune your segmentation head with.
b) Set the path to the model checkpoint from the previous step

python main_seismic_semantic.py --dataset 'Fault' --epochs 50 --batch_size 8 --frozen_weights 1 --annotator 'novice01' --model 'resnet18' --ckpt './save/SupCon/Fault_Strip_models/SimCLR_Fault_resnet18_lr_10_0.01_SimCLR_decay_0.0001_bsz_64_temp_0.07_trial_0/ckpt_epoch_100.pth'


## Links

**Associated Website**: https://ghassanalregib.info/

**Code Acknowledgement**: Code draws partially from following codebases:

* https://github.com/ultralytics/yolov5
* https://github.com/facebookresearch/detectron2


