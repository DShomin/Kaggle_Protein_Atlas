from conf import *

import gc
import os 
import argparse
import sys
import time
import cv2
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import KFold, StratifiedKFold

import torch
import torch.nn as nn
from torchvision import transforms

from dataloader import *
from models import *
from trainer import *
from transforms import *
from optimizer import *
from utils import seed_everything, find_th

import warnings
warnings.filterwarnings('ignore')


def main():

    # fix seed for train reproduction
    seed_everything(args.SEED)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("device", device)


    # TODO dataset loading


    # TODO sampling dataset for debugging
    if args.DEBUG: 
        total_num = 100
        # image_path = image_path[:total_num]
        # labels = labels[:total_num]

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.SEED)
    for fold_num, (trn_idx, val_idx) in enumerate(skf.split(image_path, labels)):

        print(f"fold {fold_num} training starts...")
        trn_img_paths = np.array(image_path)[trn_idx]
        trn_labels = np.array(labels)[trn_idx]
        val_img_paths = np.array(image_path)[val_idx]
        val_labels = np.array(labels)[val_idx]

        default_transforms = transforms.Compose([transforms.Resize(args.input_size)])
        train_transforms = get_transform(target_size=(args.input_size, args.input_size),
                                        transform_list=args.train_augments, 
                                        augment_ratio=args.augment_ratio)
                                
        valid_transforms = get_transform(target_size=(args.input_size, args.input_size),
                                        transform_list=args.valid_augments, 
                                        augment_ratio=args.augment_ratio,
                                        is_train=False)  

        train_dataset = PathDataset(trn_img_paths, trn_labels, default_transforms, train_transforms)
        valid_dataset = PathDataset(trn_img_paths, trn_labels, default_transforms, valid_transforms)
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True)

        # define model
        model = build_model(args, device)

        # optimizer definition
        optimizer = build_optimizer(args, model)
        scheduler = build_scheduler(args, optimizer, len(train_loader))
        criterion = nn.BCELoss()

        trn_cfg = {'train_loader':train_loader,
                    'valid_loader':valid_loader,
                    'model':model,
                    'criterion':criterion,
                    'optimizer':optimizer,
                    'scheduler':scheduler,
                    'device':device,
                    }

        train(args, trn_cfg)

        del model, train_loader, valid_loader, train_dataset, valid_dataset
        gc.collect()
            

if __name__ == '__main__':
    print(args)
    main()
    
