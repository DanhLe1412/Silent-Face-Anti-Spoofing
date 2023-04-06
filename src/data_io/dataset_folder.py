# -*- coding: utf-8 -*-
# @Time : 20-6-4 下午4:04
# @Author : zhuying
# @Company : Minivision
# @File : dataset_folder.py
# @Software : PyCharm

import cv2
import torch
from torchvision import datasets
import numpy as np
import os
import pandas as pd

def opencv_loader(path):
    img = cv2.imread(path)
    return img


class DatasetFolderFT(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None,
                 ft_width=10, ft_height=10, loader=opencv_loader):
        super(DatasetFolderFT, self).__init__(root, transform, target_transform, loader)
        self.root = root
        self.ft_width = ft_width
        self.ft_height = ft_height

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        # generate the FT picture of the sample
        ft_sample = generate_FT(sample)
        if sample is None:
            print('image is None --> ', path)
        if ft_sample is None:
            print('FT image is None -->', path)
        assert sample is not None

        ft_sample = cv2.resize(ft_sample, (self.ft_width, self.ft_height))
        ft_sample = torch.from_numpy(ft_sample).float()
        ft_sample = torch.unsqueeze(ft_sample, 0)

        if self.transform is not None:
            try:
                sample = self.transform(sample)
            except Exception as err:
                print('Error Occured: %s' % err, path)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, ft_sample, target


class CelebACroppedFTDataset(torch.utils.data.Dataset):
    def __init__(self, root, set_type='train', transform=None, 
                ft_width=10, ft_height=10, loader=opencv_loader):
        self.root_dir = root
        self.set_type=set_type
        self.images = self._get_all_image()
        self.ft_width = ft_width
        self.ft_height = ft_height
        self.loader = loader
        self.transform = transform

    def __len__(self):
        return len(self.images)
    
    def _get_all_image(self):
        ids = os.listdir(os.path.join(self.root_dir, self.set_type))
        images = []
        not_exsists = [0,0]
        for id in ids:
            try:
                lives = [os.path.join(self.root_dir, self.set_type, id, 'live', img) for img in os.listdir(os.path.join(self.root_dir, self.set_type, id, 'live'))]
                spoofs = [os.path.join(self.root_dir, self.set_type, id, 'spoof', img) for img in os.listdir(os.path.join(self.root_dir, self.set_type, id, 'spoof'))]
            except:
                if not os.path.exists(os.path.join(self.root_dir, self.set_type, id, 'live')):
                    not_exsists[0] += 1
                    print(os.path.join(self.root_dir, self.set_type, id, 'live'), "not exists")
                
                if not os.path.exists(os.path.join(self.root_dir, self.set_type, id, 'spoof')):
                    not_exsists[1] += 1
                    print(os.path.join(self.root_dir, self.set_type, id, 'spoof'), "not exists")
                lives = []
                spoofs = []
            images += lives + spoofs
        
        print(f"Not exists: {not_exsists[0]} lives -  {not_exsists[1]} spoofs")
        return sorted(images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        target = 1 if image_path.split("/")[-2] == "live" else 0
        sample = self.loader(image_path)
        ft_sample = generate_FT(sample)
        if sample is None:
            print('image is None --> ', image_path)
        if ft_sample is None:
            print('FT image is None -->', image_path)
        assert sample is not None
        ft_sample = cv2.resize(ft_sample, (self.ft_width, self.ft_height))
        ft_sample = torch.from_numpy(ft_sample).float()
        ft_sample = torch.unsqueeze(ft_sample, 0)

        if self.transform is not None:
            try:
                sample = self.transform(sample)
            except Exception as err:
                print('Error Occured: %s' % err, image_path)
        return sample, ft_sample, target


def generate_FT(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    fimg = np.log(np.abs(fshift)+1)
    maxx = -1
    minn = 100000
    for i in range(len(fimg)):
        if maxx < max(fimg[i]):
            maxx = max(fimg[i])
        if minn > min(fimg[i]):
            minn = min(fimg[i])
    fimg = (fimg - minn+1) / (maxx - minn+1)
    return fimg
