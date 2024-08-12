import torch.utils.data as data
import os
from PIL import Image
import pandas as pd
import scipy.io
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt


class Fault_Segmentation_Dataset(data.Dataset):
    def __init__(self, img_dir,label_dir, transform,transform_target):

        self.img_dir = img_dir
        self.label_dir = label_dir

        self.image_names = os.listdir(label_dir)
        self.transforms = transform
        self.transform_target = transform_target

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):

        img_path = os.path.join(self.img_dir,self.image_names[idx])
        label_path = os.path.join(self.label_dir,self.image_names[idx])

        im = Image.open(img_path).convert('RGB')
        target = np.asarray(Image.open(label_path))
        target = torch.from_numpy(target)





        image = self.transforms(im)
        label = self.transform_target(target)

        return image, label

class Fault_Image_Dataset(data.Dataset):
    def __init__(self, img_dir, transform):

        self.img_dir = img_dir
        self.images = os.listdir(self.img_dir)
        self.transforms = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img_path = os.path.join(self.img_dir,self.images[idx])

        im = Image.open(img_path).convert('RGB')

        image = self.transforms(im)
        return image

