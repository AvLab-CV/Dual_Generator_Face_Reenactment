"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os
from os.path import join as ospj
import json
import glob
from shutil import copyfile

from tqdm import tqdm
import ffmpeg

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.utils as vutils
import cv2

from torchvision import transforms


def save_json(json_file, filename):
    with open(filename, 'w') as f:
        json.dump(json_file, f, indent=4, sort_keys=False)


def print_network(network, name):
    num_params = 0
    for p in network.parameters():
        num_params += p.numel()
    # print(network)
    print("Number of parameters of %s: %i" % (name, num_params))


def he_init(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def denormalize(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)


def save_image(x, ncol, filename, loss = 'perceptual'):
    if loss == 'arcface':
        x = denormalize(x)
    vutils.save_image(x.cpu(), filename, nrow=ncol, padding=0)


def show_map_one(landmark):
    # N, C = landmark.size()
    # print(C)
    # print(N)
    # print(landmark.shape)
    # assert False
    # img = np.ones((3, 256, 256))
    # img = torch.from_numpy(img)
    # for i in range(0, N):
    img_3 = np.zeros((256, 256, 3))
    line_color = (255, 255, 255)
    line_width = 1
    lm_x = []
    lm_y = []
    for num in range(68):
        lm_x.append(landmark[2 * num] * 256)
        lm_y.append(landmark[2 * num + 1] * 256)
    # print(lm_x)
    # print(lm_y)
    for n in range(0, 16):
        cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
                 (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
    for n in range(17, 21):
        cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
                 (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
    for n in range(22, 26):
        cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
                 (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
    for n in range(27, 30):
        cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
                 (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
    for n in range(31, 35):
        cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
                 (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
    for n in range(36, 41):
        cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
                 (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
    cv2.line(img_3, (int(float(lm_x[36])), int(float(lm_y[36]))),
             (int(float(lm_x[41])), int(float(lm_y[41]))), line_color, line_width)
    for n in range(42, 47):
        cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
                 (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
    cv2.line(img_3, (int(float(lm_x[42])), int(float(lm_y[42]))),
             (int(float(lm_x[47])), int(float(lm_y[47]))), line_color, line_width)
    for n in range(48, 59):
        cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
                 (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
    cv2.line(img_3, (int(float(lm_x[48])), int(float(lm_y[48]))),
             (int(float(lm_x[59])), int(float(lm_y[59]))), line_color, line_width)
    for n in range(60, 67):
        cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
                 (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
    cv2.line(img_3, (int(float(lm_x[60])), int(float(lm_y[60]))),
             (int(float(lm_x[67])), int(float(lm_y[67]))), line_color, line_width)

    tensor = transforms.ToTensor()(img_3)

    # img[i, :, :, :] = tensor
        # print(img.shape)
        # assert False
    img = tensor.type(torch.cuda.FloatTensor)

    return img

def show_lm_one(landmark):
    # N, C = landmark.size()
    # print(C)
    # print(N)
    # print(landmark.shape)
    # assert False
    # img = np.ones((3, 256, 256))
    # img = torch.from_numpy(img)
    # for i in range(0, N):
    img_3 = np.zeros((256, 256, 3))
    line_color = (255, 255, 255)
    point_size = 3
    lm_x = []
    lm_y = []
    for num in range(68):
        lm_x.append(landmark[2 * num] * 256)
        lm_y.append(landmark[2 * num + 1] * 256)
    for n in range(0, 68):
        cv2.circle(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))), point_size, line_color, -1)


    tensor = transforms.ToTensor()(img_3)

    # img[i, :, :, :] = tensor
        # print(img.shape)
        # assert False
    img = tensor.type(torch.cuda.FloatTensor)

    return img



def show_lm_point(landmark):
    N, C= landmark.size()
    # print(C)
    # print(N)
    # print(landmark.shape)
    # assert False
    img = np.ones((N,3,256, 256))
    img =torch.from_numpy(img)
    for i in range(0, N):
        img_3 = np.zeros((256, 256, 3))
        line_color = (255, 255, 255)
        point_size = 2
        lm_x = []
        lm_y = []
        for num in range(68):
            lm_x.append(landmark[i, 2*num]*256)
            lm_y.append(landmark[i, 2*num+1]*256)
        # print(lm_x)
        # print(lm_y)
        for n in range(0, 68):
            cv2.circle(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))), point_size, line_color, -1)
        tensor = transforms.ToTensor()(img_3)

        img[i,:,:,:] = tensor
        # print(img.shape)
        # assert False
    img = img.type(torch.cuda.FloatTensor)


    return img

def show_map(landmark):
    N, C= landmark.size()
    # print(C)
    # print(N)
    # print(landmark.shape)
    # assert False
    img = np.ones((N,3,256, 256))
    img =torch.from_numpy(img)
    for i in range(0, N):
        img_3 = np.zeros((256, 256, 3))
        line_color = (255, 255, 255)
        line_width = 1
        lm_x = []
        lm_y = []
        for num in range(68):
            lm_x.append(landmark[i, 2*num]*256)
            lm_y.append(landmark[i, 2*num+1]*256)
        # print(lm_x)
        # print(lm_y)
        for n in range(0, 16):
            cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
                        (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
        for n in range(17, 21):
            cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
                        (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
        for n in range(22, 26):
            cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
                        (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
        for n in range(27, 30):
            cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
                        (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))),line_color, line_width)
        for n in range(31, 35):
            cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
                        (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
        for n in range(36, 41):
            cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
                        (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
        cv2.line(img_3, (int(float(lm_x[36])), int(float(lm_y[36]))),
                    (int(float(lm_x[41])), int(float(lm_y[41]))), line_color, line_width)
        for n in range(42, 47):
            cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
                        (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
        cv2.line(img_3, (int(float(lm_x[42])), int(float(lm_y[42]))),
                    (int(float(lm_x[47])), int(float(lm_y[47]))),line_color, line_width)
        for n in range(48, 59):
            cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
                        (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
        cv2.line(img_3, (int(float(lm_x[48])), int(float(lm_y[48]))),
                    (int(float(lm_x[59])), int(float(lm_y[59]))), line_color, line_width)
        for n in range(60, 67):
            cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
                        (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
        cv2.line(img_3, (int(float(lm_x[60])), int(float(lm_y[60]))),
                    (int(float(lm_x[67])), int(float(lm_y[67]))), line_color, line_width)
        # for n in range(0, 16):
        #     cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
        #                 (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), (0, 0, 0), line_width)
        # for n in range(17, 21):
        #     cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
        #                 (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), (255, 0, 0), line_width)
        # for n in range(22, 26):
        #     cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
        #                 (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), (255, 0, 0), line_width)
        # for n in range(27, 30):
        #     cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
        #                 (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), (0, 255, 0), line_width)
        # for n in range(31, 35):
        #     cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
        #                 (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), (0, 255, 0), line_width)
        # for n in range(36, 41):
        #     cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
        #                 (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), (0, 20, 125), line_width)
        # cv2.line(img_3, (int(float(lm_x[36])), int(float(lm_y[36]))),
        #             (int(float(lm_x[41])), int(float(lm_y[41]))), (0, 20, 125), line_width)
        # for n in range(42, 47):
        #     cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
        #                 (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), (0, 20, 125), line_width)
        # cv2.line(img_3, (int(float(lm_x[42])), int(float(lm_y[42]))),
        #             (int(float(lm_x[47])), int(float(lm_y[47]))), (0, 20, 125), line_width)
        # for n in range(48, 59):
        #     cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
        #                 (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), (0, 0, 255), line_width)
        # cv2.line(img_3, (int(float(lm_x[48])), int(float(lm_y[48]))),
        #             (int(float(lm_x[59])), int(float(lm_y[59]))), (0, 0, 255), line_width)
        # for n in range(60, 67):
        #     cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
        #                 (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), (255, 0, 255), line_width)
        # cv2.line(img_3, (int(float(lm_x[60])), int(float(lm_y[60]))),
        #             (int(float(lm_x[67])), int(float(lm_y[67]))), (255, 0, 255), line_width)
        # cv2.imshow('1', img_3)
        # cv2.waitKey(0)
        tensor = transforms.ToTensor()(img_3)

        img[i,:,:,:] = tensor
        # print(img.shape)
        # assert False
    img = img.type(torch.cuda.FloatTensor)


    return img

def turn_eye(fake_lm, x2_label):
    N, C = fake_lm.size()

    real_left_eye_dis = torch.abs(((x2_label[:, 2 * 36:2 * 36 + 1] - x2_label[:, 2 * 39:2 * 39 + 1]) ** 2 + (
            x2_label[:, 2 * 36 + 1:2 * 36 + 2] - x2_label[:, 2 * 39 + 1:2 * 39 + 2]) ** 2) ** (0.5))

    fake_left_eye_dis = torch.abs(((fake_lm[:, 2 * 36:2 * 36 + 1] - fake_lm[:, 2 * 39:2 * 39 + 1]) ** 2 + (
            fake_lm[:, 2 * 36 + 1:2 * 36 + 2] - fake_lm[:, 2 * 39 + 1:2 * 39 + 2]) ** 2) ** (0.5))

    ratio = fake_left_eye_dis/real_left_eye_dis
    # print(ratio)
    # assert False

    real_left_eye_1_x = x2_label[:, 2 * 37:2 * 37 + 1] - x2_label[:, 2 * 41:2 * 41 + 1]
    real_left_eye_1_y = x2_label[:, 2 * 37 + 1:2 * 37 + 2] - x2_label[:, 2 * 41 + 1:2 * 41 + 2]

    real_left_eye_2_x = x2_label[:, 2 * 38:2 * 38 + 1] - x2_label[:, 2 * 40:2 * 40 + 1]
    real_left_eye_2_y = x2_label[:, 2 * 38 + 1:2 * 38 + 2] - x2_label[:, 2 * 40 + 1:2 * 40 + 2]

    real_right_eye_1_x = x2_label[:, 2 * 43:2 * 43 + 1] - x2_label[:, 2 * 47:2 * 47 + 1]
    real_right_eye_1_y = x2_label[:, 2 * 43 + 1:2 * 43 + 2] - x2_label[:, 2 * 47 + 1:2 * 47 + 2]

    real_right_eye_2_x = x2_label[:, 2 * 44:2 * 44 + 1] - x2_label[:, 2 * 46:2 * 46 + 1]
    real_right_eye_2_y = x2_label[:, 2 * 44 + 1:2 * 44 + 2] - x2_label[:, 2 * 46 + 1:2 * 46 + 2]


    for i in range(0, N):
        # print(fake_lm[i, 2 * 37:2 * 37 + 1])
        fake_lm[i, 2 * 37:2 * 37 + 1] = fake_lm[i, 2 * 41:2 * 41 + 1] + real_left_eye_1_x[i]*ratio[i]
        # print(real_left_eye_1_x[i])
        # print(fake_lm[i, 2 * 37:2 * 37 + 1])
        # print(fake_lm[i, 2 * 37])
        # assert False
        # print(fake_lm[i, 2 * 37 + 1:2 * 37 + 2])
        fake_lm[i, 2 * 37 + 1:2 * 37 + 2] = fake_lm[i, 2 * 41 + 1:2 * 41 + 2] + real_left_eye_1_y[i]*ratio[i]
        # print(real_left_eye_1_y[i])
        # print(fake_lm[i, 2 * 37 + 1:2 * 37 + 2])
        # assert False


        fake_lm[i, 2 * 38:2 * 38 + 1] = fake_lm[i, 2 * 40:2 * 40 + 1] + real_left_eye_2_x[i]*ratio[i]
        fake_lm[i, 2 * 38 + 1:2 * 38 + 2] = fake_lm[i, 2 * 40 + 1:2 * 40 + 2] + real_left_eye_2_y[i]*ratio[i]

        fake_lm[i, 2 * 43:2 * 43 + 1] = fake_lm[i, 2 * 47:2 * 47 + 1] + real_right_eye_1_x[i]*ratio[i]
        fake_lm[i, 2 * 43 + 1:2 * 43 + 2] = fake_lm[i, 2 * 47 + 1:2 * 47 + 2] + real_right_eye_1_y[i]*ratio[i]

        fake_lm[i, 2 * 44:2 * 44 + 1] = fake_lm[i, 2 * 46:2 * 46 + 1] + real_right_eye_2_x[i]*ratio[i]
        fake_lm[i, 2 * 44 + 1:2 * 44 + 2] = fake_lm[i, 2 * 46 + 1:2 * 46 + 2] + real_right_eye_2_y[i]*ratio[i]
    return fake_lm

@torch.no_grad()
def show_lm_test(nets, args, x1_label,x2_label,x3_label,filename_real, filename_fake,filename_fake_rec, vgg_encode =None):
    # N, C, H, W = x1.size()
    N, C = x2_label.size()
    # device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    # noise = torch.randn(N, 100, device=device)
    fea_id_1 = vgg_encode(x3_label)
    # fea_id_1 = nets.id_linear_encoder(fea_id_1)
    fea_lm_1 = nets.lm_linear_encoder(x2_label)

    fake_1 = nets.linear_decoder(fea_lm_1, fea_id_1)

    fake_1_eye = turn_eye(fake_1, x2_label)


    fake_lm = show_map(fake_1_eye)


    fea_lm_2 = nets.lm_linear_encoder(x2_label)
    # fea_id_2 = nets.id_linear_encoder(x3_label)
    fea_id_2 = vgg_encode(x1_label)
    # fea_id_2 = nets.id_linear_encoder(fea_id_2)

    fake_2 = nets.linear_decoder(fea_lm_2, fea_id_2)

    # real_left_eye_1_x = x2_label[:, 2 * 37:2 * 37 + 1] - x2_label[:, 2 * 41:2 * 41 + 1]
    # real_left_eye_1_y = x2_label[:, 2 * 37 + 1:2 * 37 + 2] - x2_label[:, 2 * 41 + 1:2 * 41 + 2]
    #
    # real_left_eye_2_x = x2_label[:, 2 * 38:2 * 38 + 1] - x2_label[:, 2 * 40:2 * 40 + 1]
    # real_left_eye_2_y = x2_label[:, 2 * 38 + 1:2 * 38 + 2] - x2_label[:, 2 * 40 + 1:2 * 40 + 2]
    #
    # real_right_eye_1_x = x2_label[:, 2 * 43:2 * 43 + 1] - x2_label[:, 2 * 47:2 * 47 + 1]
    # real_right_eye_1_y = x2_label[:, 2 * 43 + 1:2 * 43 + 2] - x2_label[:, 2 * 47 + 1:2 * 47 + 2]
    #
    # real_right_eye_2_x = x2_label[:, 2 * 44:2 * 44 + 1] - x2_label[:, 2 * 46:2 * 46 + 1]
    # real_right_eye_2_y = x2_label[:, 2 * 44 + 1:2 * 44 + 2] - x2_label[:, 2 * 46 + 1:2 * 46 + 2]
    fake_2_eye = turn_eye(fake_2, x2_label)
    # print(fake_2_eye == fake_2)
    # print(fake_2[:, 2 * 44 + 1:2 * 44 + 2])
    # print(fake_2_eye[:, 2 * 44 + 1:2 * 44 + 2])
    # assert False
    # real_left_eye_1 = torch.mean(torch.abs(((x2_label[:, 2 * 37:2 * 37 + 1] - x2_label[:, 2 * 41:2 * 41 + 1]) ** 2 + (
    #             x2_label[:, 2 * 37 + 1:2 * 37 + 2] - x2_label[:, 2 * 41 + 1:2 * 41 + 2]) ** 2) ** (0.5)))
    # real_left_eye_2 = torch.mean(torch.abs(((x2_label[:, 2 * 38:2 * 38 + 1] - x2_label[:, 2 * 40:2 * 40 + 1]) ** 2 + (
    #         x2_label[:, 2 * 38 + 1:2 * 38 + 2] - x2_label[:, 2 * 40 + 1:2 * 40 + 2]) ** 2) ** (0.5)))
    # real_right_eye_1 = torch.mean(torch.abs(((x2_label[:, 2 * 43:2 * 43 + 1] - x2_label[:, 2 * 47:2 * 47 + 1]) ** 2 + (
    #         x2_label[:, 2 * 43 + 1:2 * 43 + 2] - x2_label[:, 2 * 47 + 1:2 * 47 + 2]) ** 2) ** (0.5)))
    # real_right_eye_2 = torch.mean(torch.abs(((x2_label[:, 2 * 44:2 * 44 + 1] - x2_label[:, 2 * 46:2 * 46 + 1]) ** 2 + (
    #         x2_label[:, 2 * 44 + 1:2 * 44 + 2] - x2_label[:, 2 * 46 + 1:2 * 46 + 2]) ** 2) ** (0.5)))

    fake_lm_2 = show_map(fake_2_eye)

    # print(x1.shape)
    # real_lm1 = show_map(x1_label)
    real_lm2 = show_map(x2_label)
    # real_lm3 = show_map(x3_label)
    # real_lm4 = show_map(x4_label)

    # fake_1 = nets.linear_decoder(fea_lm, fea_pe)
    # fake_lm = show_map(fake_1)
    #
    # fea_lm_2 = nets.lm_linear_encoder(fake_1)
    # fea_pe_2 = nets.pe_linear_encoder(x1_label[:, 0:20])
    #
    # fake_2 = nets.linear_decoder(fea_lm_2, fea_pe_2,noise)
    # fake_lm_2 = show_map(fake_2)

    # loss_pe = torch.mean(torch.abs(fake_lm - fake_lm_2))
    # print( torch.mean(torch.abs(fake_lm - fake_lm_2)))
    # print( torch.mean(torch.abs(fake_lm[2,:,:,:] - fake_lm[0,:,:,:])))
    # print( torch.mean(torch.abs(real_lm - fake_lm)))
    # print( torch.mean(torch.abs(real_lm2 - fake_lm_2)))

    # print(AUR_x1)
    # print(AUR_x2)
    # out_1, out_2, out_3 = nets.discriminator_tran(x1_lm)
    # print(out_1, out_2, out_3)
    # out_1, out_2, out_3 = nets.discriminator_tran(fake_lm)
    # print(out_1, out_2, out_3)
    # out_1, out_2, out_3 = nets.discriminator_tran(x2_target_lm)
    # print(out_1, out_2, out_3)
    # print(x1_lm.shape)
    # print(fake_lm_2.shape)
    # print(x2_target_lm.shape)
    # print(fake_lm.shape)

    # x_concat = [real_lm1,real_lm2 ,fake_lm,real_lm3,real_lm4, fake_lm_2]
    # x_concat = [real_lm2, x1_label[:, [0, 1, 2], :, :], x3_label[:, [0, 1, 2], :, :], fake_lm, fake_lm_2]
    # x_concat = [real_lm1,real_lm2 ,fake_lm,real_lm3, fake_lm_2]
    # x_concat = [real_lm2, real_lm3, real_lm4, fake_lm_2]

    # x_concat = torch.cat(x_concat, dim=0)
    # save_image(x_concat, N, filename, loss = args.loss)
    # del x_concat

    # real_lm2_point = show_lm_point(x2_label)
    # fake_lm_point = show_lm_point(fake_1_eye)
    # fake_lm_2_point = show_lm_point(fake_2_eye)

    for i  in range(N):

        save_image(real_lm2[i], 1, filename_real[:-4]+str(i) +'.jpg')
        save_image(fake_lm[i], 1, filename_fake[:-4]+str(i)+'.jpg')
        save_image(fake_lm_2[i], 1, filename_fake_rec[:-4]+str(i)+'.jpg')

        # save_image(real_lm2_point[i], 1, filename_real[:-4]+str(i) +'.jpg')
        # save_image(fake_lm_point[i], 1, filename_fake[:-4]+str(i)+'.jpg')
        # save_image(fake_lm_2_point[i], 1, filename_fake_rec[:-4]+str(i)+'.jpg')



@torch.no_grad()
def show_lm(nets, args, x1_label,x2_label,x3_label,x4_label, x1_id, x3_id,filename, vgg_encode =None):
    # N, C, H, W = x1.size()
    N, C = x2_label.size()
    # device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    # noise = torch.randn(N, 100, device=device)
    fea_id_1 = vgg_encode(x3_label)
    # fea_id_1 = nets.id_linear_encoder(fea_id_1)
    fea_lm_1 = nets.lm_linear_encoder(x2_label)

    fake_1 = nets.linear_decoder(fea_lm_1, fea_id_1)
    fake_lm = show_map(fake_1)


    fea_lm_2 = nets.lm_linear_encoder(x2_label)
    # fea_id_2 = nets.id_linear_encoder(x3_label)
    fea_id_2 = vgg_encode(x1_label)
    # fea_id_2 = nets.id_linear_encoder(fea_id_2)

    fake_2 = nets.linear_decoder(fea_lm_2, fea_id_2)
    fake_lm_2 = show_map(fake_2)

    # print(x1.shape)
    # real_lm1 = show_map(x1_label)
    real_lm2 = show_map(x2_label)
    # real_lm3 = show_map(x3_label)
    if args.dataset == 'rafd':

        real_lm4 = show_map(x4_label)

    # fake_1 = nets.linear_decoder(fea_lm, fea_pe)
    # fake_lm = show_map(fake_1)
    #
    # fea_lm_2 = nets.lm_linear_encoder(fake_1)
    # fea_pe_2 = nets.pe_linear_encoder(x1_label[:, 0:20])
    #
    # fake_2 = nets.linear_decoder(fea_lm_2, fea_pe_2,noise)
    # fake_lm_2 = show_map(fake_2)

    # loss_pe = torch.mean(torch.abs(fake_lm - fake_lm_2))
    # print( torch.mean(torch.abs(fake_lm - fake_lm_2)))
    # print( torch.mean(torch.abs(fake_lm[2,:,:,:] - fake_lm[0,:,:,:])))
    # print( torch.mean(torch.abs(real_lm - fake_lm)))
    # print( torch.mean(torch.abs(real_lm2 - fake_lm_2)))

    # print(AUR_x1)
    # print(AUR_x2)
    # out_1, out_2, out_3 = nets.discriminator_tran(x1_lm)
    # print(out_1, out_2, out_3)
    # out_1, out_2, out_3 = nets.discriminator_tran(fake_lm)
    # print(out_1, out_2, out_3)
    # out_1, out_2, out_3 = nets.discriminator_tran(x2_target_lm)
    # print(out_1, out_2, out_3)
    # print(x1_lm.shape)
    # print(fake_lm_2.shape)
    # print(x2_target_lm.shape)
    # print(fake_lm.shape)

    # x_concat = [real_lm1,real_lm2 ,fake_lm,real_lm3,real_lm4, fake_lm_2]

    if args.dataset == 'rafd':
        # print(real_lm2.shape)
        # print(real_lm4.shape)
        x_concat = [real_lm2, x1_label[:, [0, 1, 2], :, :], x3_label[:, [0, 1, 2], :, :], fake_lm, fake_lm_2, real_lm4]
        # x_concat = [real_lm2, x1_label[:, [0, 1, 2], :, :], x3_label[:, [0, 1, 2], :, :], fake_lm, fake_lm_2, real_lm4]
    else:
        x_concat = [real_lm2, x1_label[:, [0, 1, 2], :, :], x3_label[:, [0, 1, 2], :, :], fake_lm, fake_lm_2]
    # x_concat = [real_lm1,real_lm2 ,fake_lm,real_lm3, fake_lm_2]
    # x_concat = [real_lm2, real_lm3, real_lm4, fake_lm_2]

    x_concat = torch.cat(x_concat, dim=0)
    save_image(x_concat, N, filename, loss = args.loss)
    # for i  in range(32):
    #     save_image(real_lm4[i], 1, filename+str(i) +'.jpg')
    # save_image(x_fake[:,[3],:,:], N, filename+'_lm.jpg')
    # assert False
    del x_concat


@torch.no_grad()
def translate_and_reconstruct_lm(nets, args, x1,x1_lm, x2_target, x2_target_lm, filename, AUR_x1, AUR_x2):
    N, C, H, W = x1.size()
    fake_lm = nets.transformer(x1, AUR_x2)
    # print(AUR_x1)
    # print(AUR_x2)
    # out_1, out_2, out_3 = nets.discriminator_tran(x1_lm)
    # print(out_1, out_2, out_3)
    # out_1, out_2, out_3 = nets.discriminator_tran(fake_lm)
    # print(out_1, out_2, out_3)
    # out_1, out_2, out_3 = nets.discriminator_tran(x2_target_lm)
    # print(out_1, out_2, out_3)

    x_concat = [x1,x1_lm ,fake_lm,x2_target_lm, x2_target]

    x_concat = torch.cat(x_concat, dim=0)
    save_image(x_concat, N, filename, loss = args.loss)
    # save_image(x_fake[:,[3],:,:], N, filename+'_lm.jpg')
    # assert False
    del x_concat

@torch.no_grad()
def translate_and_reconstruct_sample(nets, args, x1, x2_target, x2_target_lm, filename, conf, arcface):
    N, C, H, W = x1.size()
    s_ref = nets.style_encoder(x1)
    masks = None
    x_fake = nets.generator(x2_target_lm, s_ref, masks=masks)
    # print(x_fake.shape)
    loss_id, dis = arcface.extract_fea(args, conf, x1, x_fake, False)
    loss_id_2, dis_2 = arcface.extract_fea(args, conf, x1, x2_target, False)
    print(dis)
    print(dis_2)
    # assert False

    # s_src = nets.style_encoder(x_src, y_src)
    # masks = nets.fan.get_heatmap(x_fake) if args.w_hpf > 0 else None
    # x_rec = nets.generator(x_fake, s_src, masks=masks)
    # print(x2_target_lm.shape)
    # print(x_source_4_channel[:,:3,:,:].shape)
    # print(x_fake[:,:3,:,:].shape)
    # print(x2_target.shape)
    x_concat = [x2_target_lm, x1, x_fake[:, [0, 1, 2], :, :], x2_target]
    x_concat = torch.cat(x_concat, dim=0)
    save_image(x_concat, N, filename)
    # save_image(x_fake[:,[3],:,:], N, filename+'_lm.jpg')

    del x_concat

@torch.no_grad()
def translate_and_reconstruct(nets, args, x1,x1_lm, x2_target, x2_target_lm, filename):
    N, C, H, W = x1.size()
    s_ref = nets.style_encoder(x1)
    masks = None
    if args.transformer:
        target_lm_fea = nets.lm_encoder(x2_target_lm, s_ref, masks=masks, loss_select=args.loss)
        source_lm_fea = nets.lm_encoder(x1_lm, s_ref, masks=masks, loss_select=args.loss)

        final_lm_fea = nets.lm_transformer(target_lm_fea, source_lm_fea, masks=masks, loss_select=args.loss)

        x_fake = nets.lm_decoder(final_lm_fea, s_ref, masks=masks, loss_select=args.loss)

    else:
        x_fake = nets.generator(x2_target_lm, s_ref, masks=masks, loss_select=args.loss)
    # x_fake = nets.generator(x2_target_lm, s_ref, masks=masks, loss_select=args.loss)

    # print(x_fake.shape)
    # loss_id, dis = arcface.extract_fea(args, conf, x2_target, x_fake, False)
    # loss_id_2, dis_2 = arcface.extract_fea(args, conf, x2_target, x2_target, False)
    # print(dis)
    # print(dis_2)
    # assert False

    # s_src = nets.style_encoder(x_src, y_src)
    # masks = nets.fan.get_heatmap(x_fake) if args.w_hpf > 0 else None
    # x_rec = nets.generator(x_fake, s_src, masks=masks)
    # print(x2_target_lm.shape)
    # print(x_source_4_channel[:,:3,:,:].shape)
    # print(x_fake[:,:3,:,:].shape)
    # print(x2_target.shape)
    x_concat = [x2_target_lm, x1,x1_lm,x_fake[:,[0,1,2],:,:], x2_target]

    x_concat = torch.cat(x_concat, dim=0)
    save_image(x_concat, N, filename, loss = args.loss)
    # save_image(x_fake[:,[3],:,:], N, filename+'_lm.jpg')
    # assert False
    del x_concat

@torch.no_grad()
def translate_and_reconstruct2(nets, args, x1,x1_lm, x2_target, x2_target_lm, filename):
    N, C, H, W = x1.size()
    s_ref = nets.style_encoder(x1)
    masks = None
    x_fake = nets.generator(x2_target_lm, s_ref, masks=masks)

    # print(x_fake.shape)
    # loss_id, dis = arcface.extract_fea(args, conf, x2_target, x_fake, False)
    # loss_id_2, dis_2 = arcface.extract_fea(args, conf, x2_target, x2_target, False)
    # print(dis)
    # print(dis_2)
    # assert False

    # s_src = nets.style_encoder(x_src, y_src)
    # masks = nets.fan.get_heatmap(x_fake) if args.w_hpf > 0 else None
    # x_rec = nets.generator(x_fake, s_src, masks=masks)
    # print(x2_target_lm.shape)
    # print(x_source_4_channel[:,:3,:,:].shape)
    # print(x_fake[:,:3,:,:].shape)
    # print(x2_target.shape)
    x_concat = [x_fake[:,[0,1,2],:,:]]

    x_concat = torch.cat(x_concat, dim=0)
    save_image(x_concat, N, filename)
    # save_image(x_fake[:,[3],:,:], N, filename+'_lm.jpg')
    # assert False
    del x_concat


@torch.no_grad()
def translate_using_latent(nets, args, x_src, y_trg_list, z_trg_list, psi, filename):
    N, C, H, W = x_src.size()
    latent_dim = z_trg_list[0].size(1)
    x_concat = [x_src]
    masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None

    for i, y_trg in enumerate(y_trg_list):
        z_many = torch.randn(10000, latent_dim).to(x_src.device)
        y_many = torch.LongTensor(10000).to(x_src.device).fill_(y_trg[0])
        s_many = nets.mapping_network(z_many, y_many)
        s_avg = torch.mean(s_many, dim=0, keepdim=True)
        s_avg = s_avg.repeat(N, 1)

        for z_trg in z_trg_list:
            s_trg = nets.mapping_network(z_trg, y_trg)
            s_trg = torch.lerp(s_avg, s_trg, psi)
            x_fake = nets.generator(x_src, s_trg, masks=masks)
            x_concat += [x_fake]

    x_concat = torch.cat(x_concat, dim=0)
    save_image(x_concat, N, filename)


@torch.no_grad()
def translate_using_reference(nets, args, x_src, x_ref, y_ref, filename):
    N, C, H, W = x_src.size()
    wb = torch.ones(1, C, H, W).to(x_src.device)
    x_src_with_wb = torch.cat([wb, x_src], dim=0)

    masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None
    s_ref = nets.style_encoder(x_ref, y_ref)
    s_ref_list = s_ref.unsqueeze(1).repeat(1, N, 1)
    x_concat = [x_src_with_wb]
    for i, s_ref in enumerate(s_ref_list):
        x_fake = nets.generator(x_src, s_ref, masks=masks)
        x_fake_with_ref = torch.cat([x_ref[i:i+1], x_fake], dim=0)
        x_concat += [x_fake_with_ref]

    x_concat = torch.cat(x_concat, dim=0)
    save_image(x_concat, N+1, filename)
    del x_concat


@torch.no_grad()
def debug_image(nets, args, inputs, step, vgg_encode = None):
    # x_src, y_src = inputs.x_src, inputs.y_src
    # x_ref, y_ref = inputs.x_ref, inputs.y_ref

    # x1, x2_target, x2_target_lm = inputs.x1, inputs.x2, inputs.x2_lm
    # x1_lm =  inputs.x_lm
    # x1_label = inputs.x_label
    # x2_label = inputs.x2_label
    # x1_one_hot, x2_one_hot = inputs.x1_one_hot, inputs.x2_one_hot
    #
    # param_x1 = x1_label[:, 0, :].unsqueeze(0)
    # param_x1 = param_x1.view(-1, 136).float()
    #
    #
    # param_x2 = x2_label[:, 0, :].unsqueeze(0)
    # param_x2 = param_x2.view(-1, 136).float()
    #
    # one_hot_x1 = x1_one_hot[:, 0, :].unsqueeze(0)
    # one_hot_x1 = one_hot_x1.view(-1, 150).float()
    #
    # one_hot_x2 = x2_one_hot[:, 0, :].unsqueeze(0)
    # one_hot_x2 = one_hot_x2.view(-1, 150).float()

    x1_label = inputs.x_label
    x2_label = inputs.x2_label
    x3_label = inputs.x3_label


    x1_one_hot, x3_one_hot = inputs.x1_one_hot, inputs.x3_one_hot
    x1_id, x3_id = inputs.x1_id, inputs.x3_id

    param_x1 = x1_label
    # param_x1 = x1_label[:, 0, :].unsqueeze(0)
    # param_x1 = param_x1.view(-1, 136).float()

    param_x2 = x2_label[:, 0, :].unsqueeze(0)
    param_x2 = param_x2.view(-1, 136).float()

    param_x3 = x3_label
    # param_x3 = x3_label[:, 0, :].unsqueeze(0)
    # param_x3 = param_x3.view(-1, 136).float()

    if args.dataset == 'mpie' or args.dataset == 'rafd':
        x4_label = inputs.x4_label
        param_x4 = x4_label[:, 0, :].unsqueeze(0)
        param_x4 = param_x4.view(-1, 136).float()

    elif args.dataset == 'vox1':
        param_x4 = None


    # one_hot_x1 = x1_one_hot[:, 0, :].unsqueeze(0)
    # one_hot_x1 = one_hot_x1.view(-1, 12606).float()
    #
    # one_hot_x3 = x3_one_hot[:, 0, :].unsqueeze(0)
    # one_hot_x3 = one_hot_x3.view(-1, 12606).float()




    # translate and reconstruct (reference-guided)
    filename = ospj(args.sample_dir, '%06d_cycle_consistency.jpg' % (step))
    # translate_and_reconstruct(nets, args, x1,x1_lm, x2_target, x2_target_lm, filename)
    show_lm(nets, args, param_x1,param_x2,param_x3,param_x4,x1_id, x3_id,filename, vgg_encode =vgg_encode)
    # # latent-guided image synthesis
    # y_trg_list = [torch.tensor(y).repeat(N).to(device)
    #               for y in range(min(args.num_domains, 5))]
    # z_trg_list = torch.randn(args.num_outs_per_domain, 1, args.latent_dim).repeat(1, N, 1).to(device)
    # for psi in [0.5, 0.7, 1.0]:
    #     filename = ospj(args.sample_dir, '%06d_latent_psi_%.1f.jpg' % (step, psi))
    #     translate_using_latent(nets, args, x_src, y_trg_list, z_trg_list, psi, filename)

    # # reference-guided image synthesis
    # filename = ospj(args.sample_dir, '%06d_reference.jpg' % (step))
    # translate_using_reference(nets, args, x_src, x_ref, y_ref, filename)


# ======================= #
# Video-related functions #
# ======================= #


def sigmoid(x, w=1):
    return 1. / (1 + np.exp(-w * x))


def get_alphas(start=-5, end=5, step=0.5, len_tail=10):
    return [0] + [sigmoid(alpha) for alpha in np.arange(start, end, step)] + [1] * len_tail


def interpolate(nets, args, x_src, s_prev, s_next):
    ''' returns T x C x H x W '''
    B = x_src.size(0)
    frames = []
    masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None
    alphas = get_alphas()

    for alpha in alphas:
        s_ref = torch.lerp(s_prev, s_next, alpha)
        x_fake = nets.generator(x_src, s_ref, masks=masks)
        entries = torch.cat([x_src.cpu(), x_fake.cpu()], dim=2)
        frame = torchvision.utils.make_grid(entries, nrow=B, padding=0, pad_value=-1).unsqueeze(0)
        frames.append(frame)
    frames = torch.cat(frames)
    return frames


def slide(entries, margin=32):
    """Returns a sliding reference window.
    Args:
        entries: a list containing two reference images, x_prev and x_next, 
                 both of which has a shape (1, 3, 256, 256)
    Returns:
        canvas: output slide of shape (num_frames, 3, 256*2, 256+margin)
    """
    _, C, H, W = entries[0].shape
    alphas = get_alphas()
    T = len(alphas) # number of frames

    canvas = - torch.ones((T, C, H*2, W + margin))
    merged = torch.cat(entries, dim=2)  # (1, 3, 512, 256)
    for t, alpha in enumerate(alphas):
        top = int(H * (1 - alpha))  # top, bottom for canvas
        bottom = H * 2
        m_top = 0  # top, bottom for merged
        m_bottom = 2 * H - top
        canvas[t, :, top:bottom, :W] = merged[:, :, m_top:m_bottom, :]
    return canvas

@torch.no_grad()
def video_rec(nets, args, x1_src,  x2_ref, x2_lm, fname):
    video = []
    content = nets.style_encoder(x1_src)
    x2_lm_prev = None
    for data_next in tqdm(zip(x1_src, x2_lm, content), 'video_rec', len(x1_src)):
        x1_next, x2_lm_next, content_lm_next = [d.unsqueeze(0) for d in data_next]
        if x2_lm_prev is None:
            x1_prev, x2_lm_prev, content_prev = x1_next, x2_lm_next, content_next
            continue
        if x2_prev != x2_next:
            x1_prev, x2_lm_prev, content_prev = x1_next, x2_lm_next, content_next
            continue

        interpolated = interpolate(nets, args, x1_src, x2_lm_prev, x2_lm_next)
        entries = [x1_prev, x1_next]
        slided = slide(entries)  # (T, C, 256*2, 256)
        frames = torch.cat([slided, interpolated], dim=3).cpu()  # (T, C, 256*2, 256*(batch+1))
        video.append(frames)
        # frames = tensor2ndarray255(frames)
        # print(frames.shape)
        # import cv2
        # assert False
        x_prev, y_prev, s_prev = x_next, y_next, s_next

    # append last frame 10 time
    for _ in range(1):
        video.append(frames[-1:])
    video = tensor2ndarray255(torch.cat(video))
    save_video(fname, video)


@torch.no_grad()
def video_ref(nets, args, x_src, x_ref, y_ref, fname):
    video = []
    s_ref = nets.style_encoder(x_ref, y_ref)
    s_prev = None
    for data_next in tqdm(zip(x_ref, y_ref, s_ref), 'video_ref', len(x_ref)):
        x_next, y_next, s_next = [d.unsqueeze(0) for d in data_next]
        if s_prev is None:
            x_prev, y_prev, s_prev = x_next, y_next, s_next
            continue
        if y_prev != y_next:
            x_prev, y_prev, s_prev = x_next, y_next, s_next
            continue

        interpolated = interpolate(nets, args, x_src, s_prev, s_next)
        entries = [x_prev, x_next]
        slided = slide(entries)  # (T, C, 256*2, 256)
        frames = torch.cat([slided, interpolated], dim=3).cpu()  # (T, C, 256*2, 256*(batch+1))
        video.append(frames)
        # frames = tensor2ndarray255(frames)
        # print(frames.shape)
        # import cv2
        # assert False
        x_prev, y_prev, s_prev = x_next, y_next, s_next

    # append last frame 10 time
    for _ in range(1):
        video.append(frames[-1:])
    video = tensor2ndarray255(torch.cat(video))
    save_video(fname, video)


@torch.no_grad()
def video_latent(nets, args, x_src, y_list, z_list, psi, fname):
    latent_dim = z_list[0].size(1)
    s_list = []
    for i, y_trg in enumerate(y_list):
        z_many = torch.randn(10000, latent_dim).to(x_src.device)
        y_many = torch.LongTensor(10000).to(x_src.device).fill_(y_trg[0])
        s_many = nets.mapping_network(z_many, y_many)
        s_avg = torch.mean(s_many, dim=0, keepdim=True)
        s_avg = s_avg.repeat(x_src.size(0), 1)

        for z_trg in z_list:
            s_trg = nets.mapping_network(z_trg, y_trg)
            s_trg = torch.lerp(s_avg, s_trg, psi)
            s_list.append(s_trg)

    s_prev = None
    video = []
    # fetch reference images
    for idx_ref, s_next in enumerate(tqdm(s_list, 'video_latent', len(s_list))):
        if s_prev is None:
            s_prev = s_next
            continue
        if idx_ref % len(z_list) == 0:
            s_prev = s_next
            continue
        frames = interpolate(nets, args, x_src, s_prev, s_next).cpu()
        video.append(frames)
        s_prev = s_next
    for _ in range(10):
        video.append(frames[-1:])
    video = tensor2ndarray255(torch.cat(video))
    save_video(fname, video)


def save_video(fname, images, output_fps=30, vcodec='libx264', filters=''):
    assert isinstance(images, np.ndarray), "images should be np.array: NHWC"
    num_frames, height, width, channels = images.shape
    stream = ffmpeg.input('pipe:', format='rawvideo', 
                          pix_fmt='rgb24', s='{}x{}'.format(width, height))
    stream = ffmpeg.filter(stream, 'setpts', '2*PTS')  # 2*PTS is for slower playback
    stream = ffmpeg.output(stream, fname, pix_fmt='yuv420p', vcodec=vcodec, r=output_fps)
    stream = ffmpeg.overwrite_output(stream)
    process = ffmpeg.run_async(stream, pipe_stdin=True)
    for frame in tqdm(images, desc='writing video to %s' % fname):
        process.stdin.write(frame.astype(np.uint8).tobytes())
    process.stdin.close()
    process.wait()


def tensor2ndarray255(images):
    images = torch.clamp(images * 0.5 + 0.5, 0, 1)
    return images.cpu().numpy().transpose(0, 2, 3, 1) * 255