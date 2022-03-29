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

import os
import pickle

import lmdb
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader, Dataset



def listdir(dname):
    fnames = list(chain(*[list(Path(dname).rglob('*.' + ext))
                          for ext in ['png', 'jpg', 'jpeg', 'JPG']]))
    return fnames

class DefaultDataset(data.Dataset):
    def __init__(self, root, transform=None):
        self.samples = listdir(root[0])
        self.samples.sort()

        self.samples_2 = listdir(root[1])
        self.samples_2.sort()

        self.samples_3 = listdir(root[2])
        self.samples_3.sort()
        self.transform = transform
        self.targets = None

    def __getitem__(self, index):
        fname = self.samples[index]
        fname_2 = self.samples_2[index]
        fname_3 = self.samples_3[index]
        # print(fname)
        # print(fname_2)
        # print(fname_3)

        img = Image.open(fname).convert('RGB')
        img_2 = Image.open(fname_2).convert('RGB')
        img_3 = Image.open(fname_3).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
            img_2 = self.transform(img_2)
            img_3 = self.transform(img_3)
        return img, img_2, img_3

    def __len__(self):
        return len(self.samples)

def _get_paths_from_lmdb(dataroot):
    """get image path list from lmdb meta info"""
    meta_info = pickle.load(open(os.path.join(dataroot, 'meta_info.pkl'),
                                 'rb'))
    paths = meta_info['keys']
    sizes = meta_info['resolution']
    if len(sizes) == 1:
        sizes = sizes * len(paths)
    # print(paths, sizes)
    return paths, sizes

def _read_img_lmdb(env, key, size):
    """read image from lmdb with key (w/ and w/o fixed size)
    size: (C, H, W) tuple"""
    # print(key)
    # print(size)
    with env.begin(write=False) as txn:
        buf = txn.get(key.encode('ascii'))
        # print(buf)

    img_flat = np.frombuffer(buf, dtype=np.uint8)
    # print('ff')
    C, H, W = size
    img = img_flat.reshape(H, W, C)
    return img

def _make_dataset(root, prefix=('.jpg', '.png')):
    img_path = os.path.join(root, 'Image')
    gt_path = os.path.join(root, 'Mask')
    img_list = [
        os.path.splitext(f)[0] for f in os.listdir(gt_path)
        if f.endswith(prefix[1])
    ]
    return [(os.path.join(img_path, img_name + prefix[0]),
             os.path.join(gt_path, img_name + prefix[1]))
            for img_name in img_list]

class LMDataset_2(data.Dataset):
    def __init__(self, root_1, root_2, root_3, root_4, transform=None, train_data = 'mpie',multi=False):
        # self.samples = listdir(root)
        # self.samples.sort()
        self.multi = multi
        if multi:

            self.train_data = train_data
            self.transform = transform
            self.targets = None
            self.samples = []
            self.samples2 = []
            self.samples3 = []
            self.samples4 = []
            self.samples5 = []
            with open(root) as F:

                for line in F:
                    line = line.strip('\n')
                    # print(line.split(' '))
                    self.samples.append(line.split(' ')[0])
                    self.samples2.append(line.split(' ')[1])
                    self.samples3.append(line.split(' ')[2])
                    self.samples4.append(line.split(' ')[3])
                    self.samples5.append(line.split(' ')[4])

        else:
            self.train_data = train_data
            self.transform = transform
            # self.targets = None
            # self.samples = []
            # self.samples2 = []

            self.img_root = root_1
            self.img_root_2 = root_2
            self.img_root_3 = root_3
            self.img_root_4 = root_4
            # self.paths_gt, self.sizes_gt = _get_paths_from_lmdb(self.gt_root)
            self.paths_img, self.sizes_img = _get_paths_from_lmdb(self.img_root)
            self.paths_img_2, self.sizes_img_2 = _get_paths_from_lmdb(self.img_root_2)
            # self.paths_img_3, self.sizes_img_3 = _get_paths_from_lmdb(self.img_root_3)
            self.paths_img_4, self.sizes_img_4 = _get_paths_from_lmdb(self.img_root_4)
            # self.gt_env = lmdb.open(self.gt_root, readonly=True, lock=False, readahead=False,
            #                         meminit=False)
            self.img_env = lmdb.open(self.img_root, readonly=True, lock=False, readahead=False,
                                     meminit=False)
            self.img_env_2 = lmdb.open(self.img_root_2, readonly=True, lock=False, readahead=False,
                                       meminit=False)

            # self.img_env_3 = lmdb.open(self.img_root_3, readonly=True, lock=False, readahead=False,
            #                            meminit=False)
            self.img_env_4 = lmdb.open(self.img_root_4, readonly=True, lock=False, readahead=False,
                                       meminit=False)


    def __getitem__(self, index):
        if self.multi:
            fname_folder = self.samples[index]
            fname2_folder = self.samples2[index]
            fname3_folder = self.samples3[index]
            fname4_folder = self.samples4[index]
            fname5_folder = self.samples5[index]


            # img_1 = random.choice(range(len(os.listdir(fname_folder))))
            # img_2 = random.choice(range(len(os.listdir(fname2_folder))))

            if self.train_data == 'mpie':

                # fname_lm = fname[:-4]+'_lm.jpg'
                # fname_lm2 = fname2[:-4]+'_lm.jpg'

                # fname = fname_folder
                # fname2 = fname2_folder
                # fname_lm = '/media/micky-linux/新增磁碟區/John/stargan-v2-master/data/mpie/MPIE_s101/MPIE_FOCropped_250_250_84_s101_shape' + '/' + fname.split('/')[10] + '/' +  fname.split('/')[11]
                # fname_lm2 = '/media/micky-linux/新增磁碟區/John/stargan-v2-master/data/mpie/MPIE_s101/MPIE_FOCropped_250_250_84_s101_shape' + '/' + fname2.split('/')[10] + '/' +  fname2.split('/')[11]

                fname = fname_folder
                fname2 = fname_folder
                # fname2 = '/media/micky-linux/新增磁碟區/John/stargan-v2-master/data/300VW_Dataset_2015_12_14/001/crop_256' + '/' +fname2_folder.split('/')[10]
                fname_lm = '/media/micky-linux/新增磁碟區/John/stargan-v2-master/data/mpie/MPIE_s101/MPIE_FOCropped_250_250_84_s101_shape' + '/' + \
                           fname.split('/')[10] + '/' + fname.split('/')[11]
                fname_lm2 = fname2_folder


            elif self.train_data == '300vw':

                fname_lm = fname.split('crop')[0] + 'lm' + fname.split('crop')[1]
                fname_lm2 = fname2.split('crop')[0] + 'lm' + fname2.split('crop')[1]

            elif self.train_data == 'vox1':

                # fname_lm = 'D:/database/voxceleb1/lm'+fname.split('voxceleb1')[1]
                # fname_lm2 = 'D:/database/voxceleb1/lm' + fname2.split('voxceleb1')[1]
                fname = fname_folder
                fname2 = fname2_folder
                fname3 = fname3_folder
                fname4 = fname4_folder
                fname5 = fname5_folder


                # fname_lm = fname.split('voxceleb1_crop_256')[0] + 'voxceleb1_LM_256' + \
                #            fname.split('voxceleb1_crop_256')[1]
                fname_lm5 = fname5.split('voxceleb1_crop_256')[0] + 'voxceleb1_LM_256' + \
                            fname5.split('voxceleb1_crop_256')[1]

                # print(fname)
                # print(fname2)
                # print(fname_lm)
                # print(fname_lm2)

            # print(fname.split('crop')[0]+'lm'+fname.split('crop')[1])
            # assert False
            # 300-vw
            # fname_lm = fname.split('crop')[0]+'lm'+fname.split('crop')[1]
            # fname_lm2 = fname2.split('crop')[0]+'lm'+fname2.split('crop')[1]

            # voxceleb1
            # fname_lm = 'D:/database/voxceleb1/lm'+fname.split('voxceleb1')[1]
            # fname_lm2 = 'D:/database/voxceleb1/lm' + fname2.split('voxceleb1')[1]
            # print(fname[17:20])
            # label = int(fname[17:20])-1

            # img = Image.open(fname).convert('RGB')
            # img_lm = Image.open(fname[:-4]+'_lm.jpg').convert('RGB')
            # img2 = Image.open(fname2).convert('RGB')
            # img_lm2 = Image.open(fname2[:-4]+'_lm.jpg').convert('RGB')

            img = Image.open(fname).convert('RGB')
            img2 = Image.open(fname2).convert('RGB')
            img3 = Image.open(fname3).convert('RGB')
            img4 = Image.open(fname4).convert('RGB')
            img5 = Image.open(fname5).convert('RGB')

            img_lm5 = Image.open(fname_lm5).convert('RGB')

            if self.transform is not None:
                img = self.transform(img)
                img2 = self.transform(img2)
                img3 = self.transform(img3)
                img4 = self.transform(img4)
                img5 = self.transform(img5)

                img_lm5 = self.transform(img_lm5)
                # img_lm = self.transform(img_lm)

            return img, img2, img3, img4,img5, img_lm5

        else:

            # fname_folder = self.samples[index]
            # fname2_folder = self.samples2[index]
            #
            #
            # # img_1 = random.choice(range(len(os.listdir(fname_folder))))
            # # img_2 = random.choice(range(len(os.listdir(fname2_folder))))
            #
            #
            #
            # if self.train_data == 'mpie':
            #
            #     # fname_lm = fname[:-4]+'_lm.jpg'
            #     # fname_lm2 = fname2[:-4]+'_lm.jpg'
            #
            #     # fname = fname_folder
            #     # fname2 = fname2_folder
            #     # fname_lm = '/media/micky-linux/新增磁碟區/John/stargan-v2-master/data/mpie/MPIE_s101/MPIE_FOCropped_250_250_84_s101_shape' + '/' + fname.split('/')[10] + '/' +  fname.split('/')[11]
            #     # fname_lm2 = '/media/micky-linux/新增磁碟區/John/stargan-v2-master/data/mpie/MPIE_s101/MPIE_FOCropped_250_250_84_s101_shape' + '/' + fname2.split('/')[10] + '/' +  fname2.split('/')[11]
            #
            #     fname = fname_folder
            #     fname2 = fname_folder
            #     # fname2 = '/media/micky-linux/新增磁碟區/John/stargan-v2-master/data/300VW_Dataset_2015_12_14/001/crop_256' + '/' +fname2_folder.split('/')[10]
            #     fname_lm = '/media/micky-linux/新增磁碟區/John/stargan-v2-master/data/mpie/MPIE_s101/MPIE_FOCropped_250_250_84_s101_shape' + '/' + fname.split('/')[10] + '/' +  fname.split('/')[11]
            #     fname_lm2 = fname2_folder
            #
            #
            # elif self.train_data == '300vw':
            #
            #     fname_lm = fname.split('crop')[0] + 'lm' + fname.split('crop')[1]
            #     fname_lm2 = fname2.split('crop')[0] + 'lm' + fname2.split('crop')[1]

            if self.train_data == 'vox1':

                img_path = self.paths_img[index]
                img_path_2 = self.paths_img_2[index]
                # img_path_3 = self.paths_img_3[index]
                img_path_4 = self.paths_img_4[index]

                # gt_resolution = [int(s) for s in self.sizes_gt[index].split('_')]
                img_resolution = [int(s) for s in self.sizes_img[index].split('_')]
                img_resolution_2 = [int(s) for s in self.sizes_img_2[index].split('_')]
                # img_resolution_3 = [int(s) for s in self.sizes_img_3[index].split('_')]
                img_resolution_4 = [int(s) for s in self.sizes_img_4[index].split('_')]

                # img_gt = _read_img_lmdb(self.gt_env, gt_path, gt_resolution)
                # print(img_resolution)
                img_img = _read_img_lmdb(self.img_env, img_path, img_resolution)
                img_img_2 = _read_img_lmdb(self.img_env_2, img_path_2, img_resolution_2)
                # img_img_3 = _read_img_lmdb(self.img_env_3, img_path_3, img_resolution_3)
                img_img_4 = _read_img_lmdb(self.img_env_4, img_path_4, img_resolution_4)

                if img_img.shape[-1] != 3:
                    img_img = np.repeat(img_img, repeats=3, axis=-1)
                if img_img_2.shape[-1] != 3:
                    img_img_2 = np.repeat(img_img_2, repeats=3, axis=-1)
                # if img_img_3.shape[-1] != 3:
                #     img_img_3 = np.repeat(img_img_3, repeats=3, axis=-1)
                if img_img_4.shape[-1] != 3:
                    img_img_4 = np.repeat(img_img_4, repeats=3, axis=-1)

                img_img = img_img[:, :, [2, 1, 0]]  # bgr => rgb
                img_img_2 = img_img_2[:, :, [2, 1, 0]]  # bgr => rgb
                # img_img_3 = img_img_3[:, :, [2, 1, 0]]  # bgr => rgb
                img_img_4 = img_img_4[:, :, [2, 1, 0]]  # bgr => rgb
                # img_gt = np.squeeze(img_gt, axis=2)
                # gt = Image.fromarray(img_gt, mode='L')
            img = Image.fromarray(img_img, mode='RGB')
            img_2 = Image.fromarray(img_img_2, mode='RGB')
            # img_lm = Image.fromarray(img_img_3, mode='RGB')
            img_lm2 = Image.fromarray(img_img_4, mode='RGB')

            # print(img_path)






            # img = Image.open(fname).convert('RGB')
            # img_lm = Image.open(fname_lm).convert('RGB')
            # img2 = Image.open(fname2).convert('RGB')
            # img_lm2 = Image.open(fname_lm2).convert('RGB')



            if self.transform is not None:


                img = self.transform(img)
                img2 = self.transform(img_2)
                img_lm2 = self.transform(img_lm2)
                # img_lm = self.transform(img_lm)


            return img, img2, img_lm2
            # return img, img2, img_lm, img_lm2



    def __len__(self):
        return len(self.paths_img)

class LMDataset(data.Dataset):
    def __init__(self, root, transform=None, train_data = 'mpie',multi=False):
        # self.samples = listdir(root)
        # self.samples.sort()
        self.multi = multi
        if multi:

            self.train_data = train_data
            self.transform = transform
            self.targets = None
            self.samples = []
            self.samples2 = []
            self.samples3 = []
            self.samples4 = []
            self.samples5 = []
            with open(root) as F:

                for line in F:
                    line = line.strip('\n')
                    # print(line.split(' '))
                    self.samples.append(line.split(' ')[0])
                    self.samples2.append(line.split(' ')[1])
                    self.samples3.append(line.split(' ')[2])
                    self.samples4.append(line.split(' ')[3])
                    self.samples5.append(line.split(' ')[4])

        else:
            self.train_data = train_data
            self.transform = transform
            self.targets = None
            self.samples = []
            self.samples2 = []
            with open(root) as F:

                for line in F:

                    line = line.strip('\n')
                    # print(line.split(' '))
                    self.samples.append(line.split(' ')[0])
                    self.samples2.append(line.split(' ')[1])


    def __getitem__(self, index):
        if self.multi:
            fname_folder = self.samples[index]
            fname2_folder = self.samples2[index]
            fname3_folder = self.samples3[index]
            fname4_folder = self.samples4[index]
            fname5_folder = self.samples5[index]


            # img_1 = random.choice(range(len(os.listdir(fname_folder))))
            # img_2 = random.choice(range(len(os.listdir(fname2_folder))))

            if self.train_data == 'mpie':

                # fname_lm = fname[:-4]+'_lm.jpg'
                # fname_lm2 = fname2[:-4]+'_lm.jpg'

                # fname = fname_folder
                # fname2 = fname2_folder
                # fname_lm = '/media/micky-linux/新增磁碟區/John/stargan-v2-master/data/mpie/MPIE_s101/MPIE_FOCropped_250_250_84_s101_shape' + '/' + fname.split('/')[10] + '/' +  fname.split('/')[11]
                # fname_lm2 = '/media/micky-linux/新增磁碟區/John/stargan-v2-master/data/mpie/MPIE_s101/MPIE_FOCropped_250_250_84_s101_shape' + '/' + fname2.split('/')[10] + '/' +  fname2.split('/')[11]

                fname = fname_folder
                fname2 = fname_folder
                # fname2 = '/media/micky-linux/新增磁碟區/John/stargan-v2-master/data/300VW_Dataset_2015_12_14/001/crop_256' + '/' +fname2_folder.split('/')[10]
                fname_lm = '/media/micky-linux/新增磁碟區/John/stargan-v2-master/data/mpie/MPIE_s101/MPIE_FOCropped_250_250_84_s101_shape' + '/' + \
                           fname.split('/')[10] + '/' + fname.split('/')[11]
                fname_lm2 = fname2_folder


            elif self.train_data == '300vw':

                fname_lm = fname.split('crop')[0] + 'lm' + fname.split('crop')[1]
                fname_lm2 = fname2.split('crop')[0] + 'lm' + fname2.split('crop')[1]

            elif self.train_data == 'vox1':

                # fname_lm = 'D:/database/voxceleb1/lm'+fname.split('voxceleb1')[1]
                # fname_lm2 = 'D:/database/voxceleb1/lm' + fname2.split('voxceleb1')[1]
                fname = fname_folder
                fname2 = fname2_folder
                fname3 = fname3_folder
                fname4 = fname4_folder
                fname5 = fname5_folder


                # fname_lm = fname.split('voxceleb1_crop_256')[0] + 'voxceleb1_LM_256' + \
                #            fname.split('voxceleb1_crop_256')[1]
                fname_lm5 = fname5.split('voxceleb1_crop_256')[0] + 'voxceleb1_LM_256' + \
                            fname5.split('voxceleb1_crop_256')[1]

                # print(fname)
                # print(fname2)
                # print(fname_lm)
                # print(fname_lm2)

            # print(fname.split('crop')[0]+'lm'+fname.split('crop')[1])
            # assert False
            # 300-vw
            # fname_lm = fname.split('crop')[0]+'lm'+fname.split('crop')[1]
            # fname_lm2 = fname2.split('crop')[0]+'lm'+fname2.split('crop')[1]

            # voxceleb1
            # fname_lm = 'D:/database/voxceleb1/lm'+fname.split('voxceleb1')[1]
            # fname_lm2 = 'D:/database/voxceleb1/lm' + fname2.split('voxceleb1')[1]
            # print(fname[17:20])
            # label = int(fname[17:20])-1

            # img = Image.open(fname).convert('RGB')
            # img_lm = Image.open(fname[:-4]+'_lm.jpg').convert('RGB')
            # img2 = Image.open(fname2).convert('RGB')
            # img_lm2 = Image.open(fname2[:-4]+'_lm.jpg').convert('RGB')

            img = Image.open(fname).convert('RGB')
            img2 = Image.open(fname2).convert('RGB')
            img3 = Image.open(fname3).convert('RGB')
            img4 = Image.open(fname4).convert('RGB')
            img5 = Image.open(fname5).convert('RGB')

            img_lm5 = Image.open(fname_lm5).convert('RGB')

            if self.transform is not None:
                img = self.transform(img)
                img2 = self.transform(img2)
                img3 = self.transform(img3)
                img4 = self.transform(img4)
                img5 = self.transform(img5)

                img_lm5 = self.transform(img_lm5)
                # img_lm = self.transform(img_lm)

            return img, img2, img3, img4,img5, img_lm5

        else:

            fname_folder = self.samples[index]
            fname2_folder = self.samples2[index]


            # img_1 = random.choice(range(len(os.listdir(fname_folder))))
            # img_2 = random.choice(range(len(os.listdir(fname2_folder))))



            if self.train_data == 'mpie':

                # fname_lm = fname[:-4]+'_lm.jpg'
                # fname_lm2 = fname2[:-4]+'_lm.jpg'

                # fname = fname_folder
                # fname2 = fname2_folder
                # fname_lm = '/media/micky-linux/新增磁碟區/John/stargan-v2-master/data/mpie/MPIE_s101/MPIE_FOCropped_250_250_84_s101_shape' + '/' + fname.split('/')[10] + '/' +  fname.split('/')[11]
                # fname_lm2 = '/media/micky-linux/新增磁碟區/John/stargan-v2-master/data/mpie/MPIE_s101/MPIE_FOCropped_250_250_84_s101_shape' + '/' + fname2.split('/')[10] + '/' +  fname2.split('/')[11]

                fname = fname_folder
                fname2 = fname_folder
                # fname2 = '/media/micky-linux/新增磁碟區/John/stargan-v2-master/data/300VW_Dataset_2015_12_14/001/crop_256' + '/' +fname2_folder.split('/')[10]
                fname_lm = '/media/micky-linux/新增磁碟區/John/stargan-v2-master/data/mpie/MPIE_s101/MPIE_FOCropped_250_250_84_s101_shape' + '/' + fname.split('/')[10] + '/' +  fname.split('/')[11]
                fname_lm2 = fname2_folder


            elif self.train_data == '300vw':

                fname_lm = fname.split('crop')[0] + 'lm' + fname.split('crop')[1]
                fname_lm2 = fname2.split('crop')[0] + 'lm' + fname2.split('crop')[1]

            elif self.train_data == 'vox1':

                # fname_lm = 'D:/database/voxceleb1/lm'+fname.split('voxceleb1')[1]
                # fname_lm2 = 'D:/database/voxceleb1/lm' + fname2.split('voxceleb1')[1]


                # fname = fname_folder+ '/'+ str(os.listdir(fname_folder)[img_1])
                # fname2 = fname2_folder+ '/'+ str(os.listdir(fname2_folder)[img_2])

                # small data
                # fname = fname_folder+ '/'+ str(os.listdir(fname_folder)[img_1])
                # fname2 = fname2_folder+ '/'+ str(os.listdir(fname2_folder)[img_2])


                # all data
                fname = fname_folder
                fname2 = fname2_folder

                fname_lm = fname.split('voxceleb1_crop_256')[0] + 'voxceleb1_LM_256' +fname.split('voxceleb1_crop_256')[1]
                fname_lm2 = fname2.split('voxceleb1_crop_256')[0] + 'voxceleb1_LM_256' +fname2.split('voxceleb1_crop_256')[1]

                # print(fname)
                # print(fname2)
                # print(fname_lm)
                # print(fname_lm2)


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

            img = Image.open(fname).convert('RGB')
            img_lm = Image.open(fname_lm).convert('RGB')
            img2 = Image.open(fname2).convert('RGB')
            img_lm2 = Image.open(fname_lm2).convert('RGB')



            if self.transform is not None:


                img = self.transform(img)
                img2 = self.transform(img2)
                img_lm2 = self.transform(img_lm2)
                img_lm = self.transform(img_lm)


            return img, img2, img_lm, img_lm2




    def __len__(self):
        return len(self.samples)




def get_train_loader(root, which='source', img_size=256,
                     batch_size=8, shuffle=True, prob=0.5, num_workers=4, train_data = 'mpie',multi=False):
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


    dataset = LMDataset(root, transform=transform, train_data = train_data,multi=multi)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=True,
                           drop_last=True)

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super(DataLoaderX, self).__iter__())

def get_train_loader_vgg(root_1, root_2,root_3, root_4, which='source', img_size=256,
                     batch_size=8, prob=0.5, num_workers=4, shuffle=True, train_data = 'mpie', multi=False):
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

    dataset = LMDataset_2(root_1, root_2, root_3, root_4, transform=transform, train_data = train_data,multi=multi)
    # sampler = _make_balanced_sampler(dataset.targets)
    return DataLoaderX(dataset=dataset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=num_workers,
                    pin_memory=True,
                    drop_last=True)
    # return data.DataLoader(dataset=dataset,
    #                        batch_size=batch_size,
    #                        num_workers=num_workers,
    #                        shuffle=shuffle,
    #                        pin_memory=True,
    #                        drop_last=True)

# def get_train_loader_vgg(root, which='source', img_size=256,
#                      batch_size=8, prob=0.5, num_workers=4, shuffle=True, train_data = 'mpie', multi=False):
#     print('Preparing DataLoader to fetch %s images '
#           'during the training phase...' % which)
#
#     crop = transforms.RandomResizedCrop(
#         img_size, scale=[0.8, 1.0], ratio=[0.9, 1.1])
#     rand_crop = transforms.Lambda(
#         lambda x: crop(x) if random.random() < prob else x)
#
#     transform = transforms.Compose([
#         rand_crop,
#         transforms.Resize([img_size, img_size]),
#         transforms.ToTensor()
#     ])
#
#     dataset = LMDataset(root, transform=transform, train_data = train_data,multi=multi)
#     # sampler = _make_balanced_sampler(dataset.targets)
#     return data.DataLoader(dataset=dataset,
#                            batch_size=batch_size,
#                            num_workers=num_workers,
#                            shuffle=shuffle,
#                            pin_memory=True,
#                            drop_last=True)

def get_train_loader_lightcnn(root, which='source', img_size=256,
                     batch_size=8, prob=0.5, num_workers=4, shuffle=True, train_data = 'mpie', multi=False):
    print('Preparing DataLoader to fetch %s images '
          'during the training phase...' % which)

    crop = transforms.RandomResizedCrop(
        img_size, scale=[0.8, 1.0], ratio=[0.9, 1.1])
    rand_crop = transforms.Lambda(
        lambda x: crop(x) if random.random() < prob else x)

    transform = transforms.Compose([
        rand_crop,
        transforms.Resize([img_size, img_size]),
        # transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])

    dataset = LMDataset(root, transform=transform, train_data = train_data,multi=multi)
    # sampler = _make_balanced_sampler(dataset.targets)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           num_workers=num_workers,
                           shuffle=shuffle,
                           pin_memory=True,
                           drop_last=True)

def get_eval_loader(root, img_size=256, batch_size=32,
                    imagenet_normalize=True, shuffle=True,
                    num_workers=4, drop_last=False, train_data = 'mpie',multi=False):
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
    dataset = LMDataset(root, transform=transform, train_data = train_data,multi=multi)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=True,
                           drop_last=drop_last)
def get_eval_loader_2(root, img_size=256, batch_size=32,
                    imagenet_normalize=True, shuffle=False,
                    num_workers=4, drop_last=False, train_data = 'mpie',multi=False):
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


    dataset = DefaultDataset(root, transform=transform)
    # dataset = LMDataset(root, transform=transform, train_data = train_data)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=True,
                           drop_last=drop_last)

def get_eval_loader_vgg(root, img_size=256, batch_size=32,
                    imagenet_normalize=True, shuffle=True,
                    num_workers=4, drop_last=False, train_data = 'mpie',multi=False):
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
    dataset = LMDataset(root, transform=transform, train_data = train_data,multi=multi)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=True,
                           drop_last=drop_last)

def get_eval_loader_lightcnn(root, img_size=256, batch_size=32,
                    imagenet_normalize=True, shuffle=True,
                    num_workers=4, drop_last=False, train_data = 'mpie',multi=False):
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
        # transforms.Resize([img_size, img_size]),
        # transforms.Resize([height, width]),
        # transforms.ToTensor()
        transforms.Resize([img_size, img_size]),
        transforms.Resize([height, width]),
        # transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()

    ])


    # dataset = DefaultDataset(root, transform=transform)
    dataset = LMDataset(root, transform=transform, train_data = train_data,multi=multi)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=True,
                           drop_last=drop_last)


def get_test_loader(root, img_size=256, batch_size=32,
                    shuffle=True, num_workers=4, train_data = 'mpie',multi=False):
    print('Preparing DataLoader for the generation phase...')
    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    dataset = LMDataset(root, transform=transform, train_data = train_data,multi=multi)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           num_workers=num_workers,
                           pin_memory=True)


def get_test_loader_vgg(root_1, root_2,root_3, root_4, img_size=256, batch_size=32,
                    shuffle=True, num_workers=4, train_data = 'mpie',multi=False):
    print('Preparing DataLoader for the generation phase...')
    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor()
    ])


    dataset = LMDataset_2(root_1, root_2, root_3, root_4, transform=transform, train_data = train_data,multi=multi)
    return DataLoaderX(dataset=dataset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=num_workers,
                    pin_memory=True)
    # return data.DataLoader(dataset=dataset,
    #                        batch_size=batch_size,
    #                        num_workers=num_workers,
    #                        pin_memory=True)
# def get_test_loader_vgg(root, img_size=256, batch_size=32,
#                     shuffle=True, num_workers=4, train_data = 'mpie',multi=False):
#     print('Preparing DataLoader for the generation phase...')
#     transform = transforms.Compose([
#         transforms.Resize([img_size, img_size]),
#         transforms.ToTensor()
#     ])
#
#
#     dataset = LMDataset(root, transform=transform, train_data = train_data,multi=multi)
#     return data.DataLoader(dataset=dataset,
#                            batch_size=batch_size,
#                            num_workers=num_workers,
#                            pin_memory=True)
def get_test_loader_lightcnn(root, img_size=256, batch_size=32,
                    shuffle=True, num_workers=4, train_data = 'mpie',multi=False):
    print('Preparing DataLoader for the generation phase...')

    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        # transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])


    dataset = LMDataset(root, transform=transform, train_data = train_data,multi=multi)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           num_workers=num_workers,
                           pin_memory=True)


class InputFetcher:
    def __init__(self, loader, latent_dim=16, mode='', multi=False):
        self.loader = loader
        self.latent_dim = latent_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mode = mode
        self.multi = multi

    def _fetch_inputs(self):
        if self.multi:
            try:
                x1, x2, x3, x4,x5, x5_lm = next(self.iter)
            except (AttributeError, StopIteration):
                self.iter = iter(self.loader)
                x1, x2, x3, x4, x5, x5_lm = next(self.iter)
            return x1, x2, x3, x4, x5, x5_lm
        else:
            try:
                x1, x2, x2_lm = next(self.iter)
            except (AttributeError, StopIteration):
                self.iter = iter(self.loader)
                x1, x2, x2_lm = next(self.iter)
            return x1, x2, x2_lm

            # try:
            #     x1, x2, x_lm, x2_lm = next(self.iter)
            # except (AttributeError, StopIteration):
            #     self.iter = iter(self.loader)
            #     x1, x2, x_lm, x2_lm = next(self.iter)
            # return x1, x2, x_lm, x2_lm



    def __next__(self):
        if self.multi:
            x1, x2, x3, x4, x5, x5_lm = self._fetch_inputs()

            inputs = Munch(x1=x1, x2=x2, x3=x3, x4=x4, x5=x5, x5_lm=x5_lm)

            return Munch({k: v.to(self.device)
                          for k, v in inputs.items()})
        else:
            x1, x2, x2_lm = self._fetch_inputs()

            inputs = Munch(x1=x1, x2=x2, x2_lm=x2_lm)

            return Munch({k: v.to(self.device)
                          for k, v in inputs.items()})

            # x1, x2, x_lm, x2_lm = self._fetch_inputs()
            #
            # inputs = Munch(x1=x1, x2=x2, x_lm=x_lm, x2_lm=x2_lm)
            #
            # return Munch({k: v.to(self.device)
            #               for k, v in inputs.items()})
