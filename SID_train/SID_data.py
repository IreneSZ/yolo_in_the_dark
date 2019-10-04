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
import csv
from PIL import Image
import numpy as np
from tqdm import tqdm

class SIDData(Dataset):
    def __init__(self, inputID_dir, gt_dir, ps=512):
        with open(inputID_dir, 'r') as f:
            input_IDs = f.read().splitlines()
            self.inputIDs = list(filter(lambda x: len(x) > 0, input_IDs))
            
        self.inputID_dir = inputID_dir

    def __len__(self):
        return len(self.inputIDs)
    
    def __getitem__(self, index):
        id = self.inputIDs[index]
        output_id = id[5:]
        input_path = './data/img_dark/' + id
        output_path = './data/img_cropped' + output_id
        
        input_img = io.imread(input_path)
        output_img = io.imread(output_path)
        
        return input_img, output_img