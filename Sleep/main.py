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
from utils import seed_everything, find_th, LabelSmoothingLoss

import warnings
warnings.filterwarnings('ignore')

from glob import glob

def main():

    # fix seed for train reproduction
    seed_everything(args.SEED)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("device", device)


    # TODO dataset loading
    # glob setting
    # image_path = sorted(glob('../../polysom/sample_images/*'))
    # labels = pd.read_csv("../../polysom/train.csv")['labels'].values

    # TODO sampling dataset for debugging
    if args.DEBUG: 
        total_num = 100
        image_path = image_path[:total_num]
        labels = labels[:total_num]

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.SEED)
    for fold_num, (trn_idx, val_idx) in enumerate(skf.split(image_path, labels)):

        print(f"fold {fold_num} training starts...")
        trn_img_paths = np.array(image_path)[trn_idx]
        trn_labels = np.array(labels)[trn_idx]
        val_img_paths = np.array(image_path)[val_idx]
        val_labels = np.array(labels)[val_idx]

        # train_transforms = get_transform(target_size=(args.input_size),
        #                                 transform_list=args.train_augments)
        # valid_transforms = get_transform(target_size=(args.input_size),
        #                                 transform_list=args.train_augments,
        #                                 is_train=False)

        train_transforms = create_train_transforms(args.input_size)
        valid_transforms = create_val_transforms(args.input_size)

        train_dataset = SleepDataset(trn_img_paths, trn_labels, train_transforms, masking='soft', is_test=False)
        valid_dataset = SleepDataset(trn_img_paths, trn_labels, valid_transforms, masking='soft', is_test=False)

        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True)

        # define model
        model = build_model(args, device)

        # optimizer definition
        optimizer = build_optimizer(args, model)
        scheduler = build_scheduler(args, optimizer, len(train_loader))
        criterion = nn.CrossEntropyLoss()


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
    
