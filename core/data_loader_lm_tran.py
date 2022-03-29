"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

from pathlib import Path
from itertools import chain
import os
import random

from munch import Munch
from PIL import Image
import numpy as np

import torch
from torch.utils import data
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
import cv2
import pandas as pd
import math

# def pose_to_label():

def load_lm(path):
    label_path = path
    label = pd.read_csv(label_path)
    # a = list(range(293, 296)) + list(range(694, 711))  # + list(range(299,435))

    a = list(range(293, 296)) + list(range(676, 693))  + list(range(296, 364)) + list(range(364, 432))
    # b = list(range(296, 364)) + list(range(364, 432))
    refdf = label[label.columns[a]]
    # lm = label[label.columns[b]]
    paramA = torch.FloatTensor(refdf.values[0]).view(1, -1)
    paramA[0, 0:3] = (paramA[0, 0:3] - (-(math.pi/2))) / math.pi
    paramA[0, 3:20] = paramA[0, 3:20] / 5

    # paramB = torch.FloatTensor(refdf.values[0]).view(1, -1)
    paramA[0, 20:] = paramA[0, 20:] / 256



    return paramA


def load_label(path):
    label_path = path
    label = pd.read_csv(label_path)
    # a = list(range(293, 296)) + list(range(694, 711))  # + list(range(299,435))
    a = list(range(293, 296)) + list(range(676, 693))  # + list(range(299,435))
    refdf = label[label.columns[a]]
    paramA = torch.FloatTensor(refdf.values[0]).view(1, -1)
    paramA[0, 0:3] = (paramA[0, 0:3] - (-(math.pi/2))) / math.pi
    paramA[0, 3:20] = paramA[0, 3:20] / 5
    return paramA

def load_csv(path):
    # label_path = path
    label = pd.read_csv(path)
    landmarks = label.iloc[0, 3:].values
    # print(type(landmarks))
    # print(landmarks.shape)
    # paramA = torch.FloatTensor(label.iloc[0, 3:].values)
    paramA = torch.from_numpy(landmarks.astype(float))
    paramA = paramA.view(1, -1)
    paramA = paramA / 250
    # print(paramA)
    return paramA

def listdir(dname):
    fnames = list(chain(*[list(Path(dname).rglob('*.' + ext))
                          for ext in ['png', 'jpg', 'jpeg', 'JPG']]))
    return fnames


class LMDataset(data.Dataset):
    def __init__(self, root, transform=None, train_data = 'mpie'):
        # self.samples = listdir(root)
        # self.samples.sort()
        self.train_data = train_data
        self.transform = transform
        self.targets = None
        self.samples = []
        self.samples2 = []
        self.samples3 = []
        self.samples4 = []
        self.samples5 = []
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5]),
        ])
        if self.train_data == 'mpie':
            with open(root) as F:

                for line in F:

                    line = line.strip('\n')
                    # print(line.split(' '))
                    self.samples.append(line.split(' ')[0])
                    self.samples2.append(line.split(' ')[1])
                    self.samples3.append(line.split(' ')[2])
                    self.samples4.append(line.split(' ')[3])
        elif self.train_data == '300vw':
            with open(root) as F:

                for line in F:
                    line = line.strip('\n')
                    # print(line.split(' '))
                    self.samples.append(line.split(' ')[0])
                    self.samples2.append(line.split(' ')[1])
                    self.samples3.append(line.split(' ')[2])
                    self.samples4.append(line.split(' ')[3])
                    self.samples5.append(line.split(' ')[4])



    def __getitem__(self, index):
        if self.train_data == 'mpie':

            fname = self.samples[index]
            fname2 = self.samples2[index]
            fname3 = self.samples3[index]
            fname4 = self.samples4[index]
        elif self.train_data == '300vw':
            fname = self.samples[index]
            fname2 = self.samples2[index]
            fname3 = self.samples4[index]



        # label_path = 'D:/database/300vw_preprocess_by_crop_256'+ '/' + fname.split('/')[3]+ '/' + fname.split('/')[5][:-4] + '.csv'
        # label_path2 = 'D:/database/300vw_preprocess_by_crop_256' + '/' + fname2.split('/')[3] + '/' + fname2.split('/')[5][
        #                                                                                             :-4] + '.csv'
        #
        # fname_label = load_label(label_path)
        # fname_label2 = load_label(label_path2)


        if self.train_data == 'mpie':
            # fname_lm = fname[:-4]+'_lm.jpg'
            # fname_lm2 = fname2[:-4]+'_lm.jpg'

            # print('D:/database/MPIE/MPIE_FOCropped_250_250_84_s101_lm' + fname.split('s101')[1])
            # print('D:/database/MPIE/MPIE_FOCropped_250_250_84_s101_lm' + fname2.split('s101')[1])
            # fname_lm = 'D:/database/MPIE/MPIE_FOCropped_250_250_84_s101_lm' + fname.split('s101')[1]
            # fname_lm2 = 'D:/database/MPIE/MPIE_FOCropped_250_250_84_s101_lm' + fname2.split('s101')[1]
            # assert False

            label_path = 'D:/database/MPIE/MPIE_FOCropped_250_250_84_s101_lm_csv' + fname.split('s101')[1][:-4] + '.csv'

            label_path2 = 'D:/database/MPIE/MPIE_FOCropped_250_250_84_s101_lm_csv' + fname2.split('s101')[1][:-4] + '.csv'

            label_path3 = 'D:/database/MPIE/MPIE_FOCropped_250_250_84_s101_lm_csv' + fname3.split('s101')[1][:-4] + '.csv'

            label_path4 = 'D:/database/MPIE/MPIE_FOCropped_250_250_84_s101_lm_csv' + fname4.split('s101')[1][:-4] + '.csv'

            # print(label_path)
            # print(label_path2)

            fname_label = load_csv(label_path)
            fname_label2 = load_csv(label_path2)
            fname_label3 = load_csv(label_path3)
            fname_label4 = load_csv(label_path4)
            # assert False

            label = fname.split('/')[4]
            x1_id = [int(label)-1]
            x1_id = torch.LongTensor(x1_id)
            fname_one_hot = torch.zeros(1,150).scatter_(1, x1_id.unsqueeze(1),1)


            label_3 = fname3.split('/')[4]
            x3_id = [int(label_3)-1]

            x3_id = torch.LongTensor(x3_id)
            fname_one_hot_3 = torch.zeros(1, 150).scatter_(1, x3_id.unsqueeze(1),1)

            # assert False








        elif self.train_data == '300vw':

            label_path = fname.split('crop_256')[0] + '/csv_256/' + fname.split('crop_256')[1][:-4] + '.csv'

            label_path2 = fname2.split('crop_256')[0] + '/csv_256/' + fname2.split('crop_256')[1][:-4] + '.csv'

            label_path3 = fname3.split('crop_256')[0] + '/csv_256/' + fname3.split('crop_256')[1][:-4] + '.csv'

            # label_path4 = 'D:/database/300VW_Dataset_2015_12_14/001/csv_256' + fname4.split('crop_256')[1][:-4] + '.csv'

            # print(label_path)
            # print(label_path2)

            fname_label = load_csv(label_path)
            fname_label2 = load_csv(label_path2)
            fname_label3 = load_csv(label_path3)
            # fname_label4 = load_csv(label_path4)
            # print(fname_label)
            # assert False

            label = self.samples3[index]
            x1_id = [int(label)]
            x1_id = torch.LongTensor(x1_id)
            fname_one_hot = torch.zeros(1,100).scatter_(1, x1_id.unsqueeze(1),1)


            label_3 = self.samples5[index]
            x3_id = [int(label_3)]

            x3_id = torch.LongTensor(x3_id)
            fname_one_hot_3 = torch.zeros(1, 100).scatter_(1, x3_id.unsqueeze(1),1)
            # print(label, label_3)
            # assert False



            # label_path = './data/300vw_preprocess_by_crop_256' + '/' + fname.split('/')[3] + '/' + \
            #              fname.split('/')[5][:-4] + '.csv'
            # label_path2 = './data/300vw_preprocess_by_crop_256' + '/' + fname2.split('/')[3] + '/' + \
            #               fname2.split('/')[5][:-4] + '.csv'
            #
            #
            # fname_label = load_lm(label_path)
            # fname_label2 = load_lm(label_path)





            # print(fname_label)
            #             # assert False

            # fname_lm = fname.split('crop')[0] + 'lm' + fname.split('crop')[1]
            # fname_lm2 = fname2.split('crop')[0] + 'lm' + fname2.split('crop')[1]

        elif self.train_data == 'vox1':

            fname_lm = 'D:/database/voxceleb1/lm'+fname.split('voxceleb1')[1]
            fname_lm2 = 'D:/database/voxceleb1/lm' + fname2.split('voxceleb1')[1]




        # print(fname.split('crop')[0]+'lm'+fname.split('crop')[1])
        # assert False
        #300-vw
        # fname_lm = fname.split('crop')[0]+'lm'+fname.split('crop')[1]
        # fname_lm2 = fname2.split('crop')[0]+'lm'+fname2.split('crop')[1]


        #voxceleb1
        # fname_lm = 'D:/database/voxceleb1/lm'+fname.split('voxceleb1')[1]
        # fname_lm2 = 'D:/database/voxceleb1/lm' + fname2.split('voxceleb1')[1]
        # print(fname[17:20])
        # label = int(fname[17:20])-1

        # img = Image.open(fname).convert('RGB')
        # img_lm = Image.open(fname[:-4]+'_lm.jpg').convert('RGB')
        # img2 = Image.open(fname2).convert('RGB')
        # img_lm2 = Image.open(fname2[:-4]+'_lm.jpg').convert('RGB')

        # img = Image.open(fname).convert('RGB')
        # img_lm = Image.open(fname_lm).convert('RGB')
        # img2 = Image.open(fname2).convert('RGB')
        # img_lm2 = Image.open(fname_lm2).convert('RGB')



        # if self.transform is not None:
        #
        #
        #     img = self.transform(img)
        #     img2 = self.transform(img2)
        #     img_lm2 = self.transform(img_lm2)
        #     img_lm = self.transform(img_lm)
        if self.train_data == 'mpie':
            # print(fname_label, fname_label2, fname_label3, fname_label4, fname_one_hot, fname_one_hot_3, x1_id, x3_id)
            return fname_label, fname_label2, fname_label3, fname_label4, fname_one_hot, fname_one_hot_3, x1_id, x3_id

        elif self.train_data == '300vw':

            return fname_label, fname_label2,fname_label3, fname_one_hot, fname_one_hot_3, x1_id, x3_id



    def __len__(self):
        return len(self.samples)




def get_train_loader(root, which='source', img_size=256,
                     batch_size=8, shuffle=True, prob=0.5, num_workers=4, train_data = 'mpie'):
    print('Preparing DataLoader to fetch %s images '
          'during the training phase...' % which)

    crop = transforms.RandomResizedCrop(
        img_size, scale=[0.8, 1.0], ratio=[0.9, 1.1])
    rand_crop = transforms.Lambda(
        lambda x: crop(x) if random.random() < prob else x)

    transform = transforms.Compose([
        rand_crop,
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])


    dataset = LMDataset(root, transform=transform, train_data = train_data)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=True,
                           drop_last=True)


def get_train_loader_vgg(root, which='source', img_size=256,
                     batch_size=8, prob=0.5, num_workers=4, shuffle=True, train_data = 'mpie'):
    print('Preparing DataLoader to fetch %s images '
          'during the training phase...' % which)

    crop = transforms.RandomResizedCrop(
        img_size, scale=[0.8, 1.0], ratio=[0.9, 1.1])
    rand_crop = transforms.Lambda(
        lambda x: crop(x) if random.random() < prob else x)

    transform = transforms.Compose([
        rand_crop,
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor()
    ])

    dataset = LMDataset(root, transform=transform, train_data = train_data)
    # sampler = _make_balanced_sampler(dataset.targets)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           num_workers=num_workers,
                           shuffle=shuffle,
                           pin_memory=True,
                           drop_last=True)

def get_eval_loader(root, img_size=256, batch_size=32,
                    imagenet_normalize=True, shuffle=True,
                    num_workers=4, drop_last=False, train_data = 'mpie'):
    print('Preparing DataLoader for the evaluation phase...')
    if imagenet_normalize:
        height, width = 299, 299
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        height, width = img_size, img_size
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.Resize([height, width]),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


    # dataset = DefaultDataset(root, transform=transform)
    dataset = LMDataset(root, transform=transform, train_data = train_data)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=True,
                           drop_last=drop_last)

def get_eval_loader_vgg(root, img_size=256, batch_size=32,
                    imagenet_normalize=True, shuffle=True,
                    num_workers=4, drop_last=False, train_data = 'mpie'):
    print('Preparing DataLoader for the evaluation phase...')
    if imagenet_normalize:
        height, width = 299, 299
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        height, width = img_size, img_size
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.Resize([height, width]),
        transforms.ToTensor()

    ])


    # dataset = DefaultDataset(root, transform=transform)
    dataset = LMDataset(root, transform=transform, train_data = train_data)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=True,
                           drop_last=drop_last)

def get_test_loader(root, img_size=256, batch_size=32,
                    shuffle=True, num_workers=4, train_data = 'mpie'):
    print('Preparing DataLoader for the generation phase...')
    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    dataset = LMDataset(root, transform=transform, train_data = train_data)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           num_workers=num_workers,
                           pin_memory=True)


def get_test_loader_vgg(root, img_size=256, batch_size=32,
                    shuffle=True, num_workers=4, train_data = 'mpie'):
    print('Preparing DataLoader for the generation phase...')
    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor()
    ])


    dataset = LMDataset(root, transform=transform, train_data = train_data)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           num_workers=num_workers,
                           pin_memory=True)


class InputFetcher_mpie:
    def __init__(self, loader, latent_dim=16, mode=''):
        self.loader = loader
        self.latent_dim = latent_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mode = mode

    def _fetch_inputs(self):
        try:
            x_label, x2_label,x3_label, x4_label, x1_one_hot, x3_one_hot, x1_id, x3_id = next(self.iter)
        except (AttributeError, StopIteration):
            self.iter = iter(self.loader)
            x_label, x2_label,x3_label, x4_label, x1_one_hot, x3_one_hot, x1_id, x3_id = next(self.iter)
        return x_label, x2_label,x3_label, x4_label, x1_one_hot, x3_one_hot, x1_id, x3_id



    def __next__(self):
        x_label, x2_label,x3_label, x4_label, x1_one_hot, x3_one_hot, x1_id, x3_id = self._fetch_inputs()

        inputs = Munch(x_label=x_label, x2_label=x2_label,x3_label=x3_label, x4_label=x4_label, x1_one_hot=x1_one_hot, x3_one_hot=x3_one_hot, x1_id=x1_id, x3_id=x3_id)

        return Munch({k: v.to(self.device)
                      for k, v in inputs.items()})


class InputFetcher_300vw:
    def __init__(self, loader, latent_dim=16, mode=''):
        self.loader = loader
        self.latent_dim = latent_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mode = mode

    def _fetch_inputs(self):
        try:
            x_label, x2_label,x3_label, x1_one_hot, x3_one_hot, x1_id, x3_id = next(self.iter)
        except (AttributeError, StopIteration):
            self.iter = iter(self.loader)
            x_label, x2_label,x3_label, x1_one_hot, x3_one_hot, x1_id, x3_id = next(self.iter)
        return x_label, x2_label,x3_label, x1_one_hot, x3_one_hot, x1_id, x3_id



    def __next__(self):
        x_label, x2_label,x3_label, x1_one_hot, x3_one_hot, x1_id, x3_id = self._fetch_inputs()

        inputs = Munch(x_label=x_label, x2_label=x2_label,x3_label=x3_label, x1_one_hot=x1_one_hot, x3_one_hot=x3_one_hot, x1_id=x1_id, x3_id=x3_id)

        return Munch({k: v.to(self.device)
                      for k, v in inputs.items()})
