#!/usr/bin/env python
# coding: utf-8

import os
import sys
import time
import datetime

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

from data import TrainDataLoader
from model import DummyModel


class Trainer:
    
    def __init__(self, epochs, ID_txt, batch_size, n_cpu):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = epochs
        self.model = DummyModel().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        
        self.dataloader = TrainDataLoader(
                            ID_txt,
                            batch_size=batch_size,
                            num_workers=n_cpu)


    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            start_time = time.time()
            for batch_i, (imgs, targets, idx) in enumerate(self.dataloader):
                self.optimizer.zero_grad()
                
                batches_done = len(self.dataloader) * epoch + batch_i
                imgs = imgs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(imgs)
                ########
                loss = outputs
                ########
                loss.backward()
                self.optimizer.step()
            print(epoch)
        











