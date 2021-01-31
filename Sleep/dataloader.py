from conf import *

import os
import cv2
import copy
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader 
from albumentations.pytorch.functional import img_to_tensor



class SleepDataset(Dataset): 
    def __init__(self, image_paths, labels=None, transforms=None, masking='soft', is_test=False): 
        self.image_paths = image_paths
        self.masking = masking
        self.labels = labels 
        # self.default_transforms = default_transforms
        self.transforms = transforms
        self.is_test = is_test

        # for image_path in self.image_paths:
            # image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # img = Image.fromarray(img)

            # if self.default_transforms is not None:
            #     img = self.default_transforms(image=img)['image']
            # self.images.append(image)

    def __len__(self): 
        return len(self.image_paths)

    def __getitem__(self, index):
        
        image_path = self.image_paths[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transforms:
            image = self.transforms(image=image)['image']

        # Masking Augmentation (Saewon)
        if self.masking == 'soft':
            mask_prob = 0.3
            rand_val = random.random()
            if rand_val < mask_prob:
                image = time_mask(image, num_masks=2)
            rand_val = random.random()
            if rand_val < mask_prob:
                image = signal_mask(image, num_masks=2)
            
        elif self.masking == 'hard':
            mask_prob = 0.5
            rand_val = random.random()
            if rand_val < mask_prob:
                image = signal_mask(time_mask(image, num_masks=2), num_masks=2)

        image = img_to_tensor(image, {"mean": [0.485, 0.456, 0.406],
                                    "std": [0.229, 0.224, 0.225]})

        if self.is_test:
            return image #torch.tensor(img, dtype=torch.float32)
        else:
            label = self.labels[index]
            return image, label#torch.tensor(img, dtype=torch.float32),\
                 #torch.tensor(self.labels[index], dtype=torch.long)


# Saewon
# 세로로 masking
def time_mask(image, T=50, num_masks=1):
    cloned = copy.deepcopy(image)
    len_spectro = cloned.shape[1]

    for i in range(0, num_masks):
        t = random.randrange(10, T)
        t_zero = random.randrange(10, len_spectro - t)

        # avoids randrange error if values are equal and range is empty
        if (t_zero == t_zero + t): 
            return cloned

        mask_end = random.randrange(t_zero, t_zero + t)
        cloned[:,t_zero:mask_end] = 0
    
    return cloned

# 가로로 masking
def signal_mask(image, S=30, num_masks=1):
    cloned = copy.deepcopy(image)
    num_mel_channels = cloned.shape[1]
    
    for i in range(0, num_masks):        
        f = random.randrange(10, S)
        f_zero = random.randrange(10, num_mel_channels - f)
        
        # avoids randrange error if values are equal and range is empty
        if (f_zero == f_zero + f): 
            return cloned

        mask_end = random.randrange(f_zero, f_zero + f) 
        cloned[f_zero:mask_end] = 0
    
    return cloned
