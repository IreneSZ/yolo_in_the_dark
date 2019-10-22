# Major changes (vs. the original implementation in the paper):
# using RGB images diretly, instead of the raw sensor data

import torch
from torch.utils.data import Dataset
import os
import numpy as np
import glob
import rawpy

import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torch.nn import init
from torchvision import transforms
from torchvision.utils import save_image
from torchsummary import summary
from skimage import io, transform
import matplotlib.pyplot as plt
import random
import numpy as np
import cv2

def get_img(img_path):
    img = cv2.imread(img_path).astype(np.float32)
    return torch.from_numpy(np.transpose(img, (2, 0, 1)) / 255)

# horizontally flip the image
# def horizontal_flip(img):
#     img_output = np.flip(img, 1)
#     return img_output   

# randomly crop a pair of images of size 360*640 from a larger image
def random_crop(img1,img2, h=360, w=640):
    h_old = img1.shape[1]
    w_old = img1.shape[2]
    
    h_max = h_old - h - 10
    w_max = w_old - w- 10
    
    #print(h_old, h_max, w_old, w_max)
    
    h_min = random.randint(10, h_max)
    w_min = random.randint(10, w_max)
    
    img1_cropped = img1[:, h_min:h_min+h, w_min:w_min+w]
    img2_cropped = img2[:, h_min:h_min+h, w_min:w_min+w]

    return img1_cropped, img2_cropped


class SIDData(Dataset):
    def __init__(self, inputID_dir):
        with open(inputID_dir, 'r') as f:
            input_IDs = f.read().splitlines()
            self.inputIDs = list(filter(lambda x: len(x) > 0, input_IDs))
            
        self.inputID_dir = inputID_dir

    def __len__(self):
        return len(self.inputIDs)
    
    def __getitem__(self, index):
        ID = self.inputIDs[index]

        input_path = './data/img_dark/' + ID   # the dark img
        output_path = './data/img_raw/' + ID   # the ground truth
        
        input_img = get_img(input_path)
        output_img = get_img(output_path)
        
#         rand = random.randint(0, 5)
#         if rand % 2 != 0: 
#             input_img = horizontal_flip(input_img)
#             output_img = horizontal_flip(output_img)
#         else:
#             pass
        
        input_img, output_img = random_crop(input_img, output_img)   # randomly crop the pair
        #print(input_img.shape, output_img.shape)
        return input_img, output_img