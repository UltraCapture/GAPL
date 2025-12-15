import argparse
import os
import csv
import torch
import torchvision.transforms as transforms
import torch.utils.data
import numpy as np
from torch.utils.data import Dataset
import sys
from PIL import Image 
import pickle
from tqdm import tqdm
from io import BytesIO
from copy import deepcopy
import random
from scipy.ndimage.filters import gaussian_filter

Proptype_dataset = list([
    dict(
        name = 'LSUN',
        path = '../datasets/CNNDetection/train/',
        label = 0,
        subfolder = True,
        samples_per_folder= 150,
        contains = '0_real'
    ),

    dict(
        name = 'ProGAN',
        path = '../datasets/CNNDetection/train/',
        label = 1,
        subfolder = True,
        contains = '1_fake',
        samples_per_folder= 100,
    ),

    dict(
        name = 'ImageNet',
        path = "../datasets/GenImage/GenImage/imagenet_ai_0419_sdv4/train/nature",
        label = 0,
        samples_per_folder = 3000,
    ),
    

    dict(
        name = 'SD1.4',
        path = "../datasets/GenImage/GenImage/imagenet_ai_0419_sdv4/train/ai",
        label = 1,
        samples_per_folder = 2000,
    ),

    dict(
        name='midjourney',
        path = "../datasets/GenImage/GenImage/imagenet_midjourney/train/ai",
        label = 1,
        samples_per_folder = 2000,
    ),
])

MEAN = {
    "imagenet":[0.485, 0.456, 0.406],
    "clip":[0.48145466, 0.4578275, 0.40821073]
}

STD = {
    "imagenet":[0.229, 0.224, 0.225],
    "clip":[0.26862954, 0.26130258, 0.27577711]
}

def recursively_read(rootdir, must_contain, exts=["png", "jpg", "JPEG", "jpeg", "bmp", "PNG"]):
    out = [] 
    for r, d, f in os.walk(rootdir):
        for file in f:
            if (file.split('.')[1] in exts)  and  (must_contain in os.path.join(r, file)):
                out.append(os.path.join(r, file))
    return out

def get_list(path, must_contain=''):
    if ".pickle" in path:
        with open(path, 'rb') as f:
            image_list = pickle.load(f)
        image_list = [item for item in image_list if must_contain in item ]
    else:
        image_list = recursively_read(path, must_contain)
    return image_list

class RealFakeDataset(Dataset):
    def __init__(self,   
                jpeg_quality=None,
                gaussian_sigma=None,
                stat_from='imagenet',
                sample_seed=42):

        self.jpeg_quality = jpeg_quality
        self.gaussian_sigma = gaussian_sigma
        self.transform = transforms.Compose([
            transforms.RandomCrop(224, pad_if_needed=True, padding_mode='constant'),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN[stat_from], std=STD[stat_from]),
        ])

        self.total_list = []
        list_len = 0
        self.labels_dict = {}
        self.categorie = []
        rng = random.Random(sample_seed)

        for entry in Proptype_dataset:
            name = entry['name']
            path = entry['path']
            label = entry['label']
            samples = 0
            samples_per_folder = entry.get('samples_per_folder', None)
            must_contain = entry.get('contains', '')
            subfolder = entry.get('subfolder', False)

            if subfolder:
                for sub in sorted(os.listdir(path)):
                    sub_path = os.path.join(path, sub)
                    if not os.path.isdir(sub_path):
                        continue
                    img_list = recursively_read(sub_path, must_contain)
                    if samples_per_folder is not None and len(img_list) > samples_per_folder:
                        img_list = rng.sample(img_list, samples_per_folder)
                    for img_path in img_list:
                        self.total_list.append(img_path)
                        self.labels_dict[img_path] = label
            else:
                img_list = recursively_read(path, must_contain)
                if samples_per_folder is not None and len(img_list) > samples_per_folder:
                    img_list = rng.sample(img_list, samples_per_folder)
                for img_path in img_list:
                    self.total_list.append(img_path)
                    self.labels_dict[img_path] = label
            samples = len(self.total_list) - list_len
            list_len = len(self.total_list)
            self.categorie.append((name, samples))
    
    def summary(self, logger=None):
        for datas in self.categorie:
            print(f"Image from {datas[0]} samples {datas[1]}")
        if logger != None:
            for datas in self.categorie:
                logger.info(f"Image from {datas[0]} samples {datas[1]}")

    def __len__(self):
        return len(self.total_list)

    def __getitem__(self, idx):
        img_path = self.total_list[idx]
        label = self.labels_dict[img_path]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, label

if __name__ == "__main__":
    ds = RealFakeDataset()
    ds.summary()