import os
import cv2
import albumentations as A
abs_path = os.path.dirname(__file__)

args = {
    "SEED":42,
    "n_folds":5,
    "epochs":20,
    "num_classes":1,
    "input_size":512,
    "batch_size":8,
    "num_workers":4,
    "model":"tf_efficientnet_b4_ns",
    "optimizer":"AdamW",
    "scheduler":"Plateau",
    "lr":"1e-4",
    "weight_decay":0.0,
    "train_augments":'random_crop, horizontal_flip, vertical_flip, random_rotate, random_grayscale',
    "valid_augments":'horizontal_flip, vertical_flip',
    "augment_ratio":0.5,
    "pretrained":False,
    "lookahead":False,
    "k_param":5,
    "alpha_param":0.5,
    "patience":3,
    "albu": False,
    "DEBUG":False,
}