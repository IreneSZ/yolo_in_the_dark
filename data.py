#!/usr/bin/env python
# coding: utf-8


import glob
import math
import os
import random
import shutil
from pathlib import Path
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


def get_img(ID):
    img_path = './image/' + ID + '.jpg'
    img = cv2.imread(img_path)
    img = img[:, :, [2,1,0]].astype(np.float32) # BGR to RGB
    ############################### /255 for normalization for now################
    img /= 255
    img = torch.from_numpy(img)
    #############################################################################
    #plt.imshow(img)
    return img

def get_label(ID):
    label_path = './bbox/' + ID + '.txt'
    box = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
    
    return box

class TrainData(Dataset):
    def __init__(self, ID_txt):
        with open(ID_txt, "r") as f:
            self.ID = list(f.read().splitlines())
    
    def __len__(self):
        return len(self.ID)
    
    def __getitem__(self, index):
        ID = self.ID[index]
        img = get_img(ID)
        label = get_label(ID)
        return img, label

def collate_fn(batch):
    img = [item[0] for item in batch]
    label = [item[1] for item in batch]
    stacked_img = torch.stack(img)
    cat_label = torch.cat(label)
    idx = []
    for i, one_img_label in enumerate(label):
        l = len(one_img_label)
        idx.extend([i] * l)
    idx = torch.LongTensor(idx).reshape(-1, 1)    

    return stacked_img, cat_label, idx

class TrainDataLoader(DataLoader):
    
    def __init__(self, ID_txt, *args, **kwargs):
        dataset = TrainData(ID_txt)
        super().__init__(dataset, *args, shuffle=True, pin_memory=True, collate_fn=collate_fn, **kwargs)

