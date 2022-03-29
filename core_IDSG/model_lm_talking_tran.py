"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import copy
import math

from munch import Munch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.wing import FAN

class HighPass(nn.Module):
    def __init__(self, w_hpf, device):
        super(HighPass, self).__init__()
        self.filter = torch.tensor([[-1, -1, -1],
                                    [-1, 8., -1],
                                    [-1, -1, -1]]).to(device) / w_hpf

    def forward(self, x):
        filter = self.filter.unsqueeze(0).unsqueeze(1).repeat(x.size(1), 1, 1, 1)
        return F.conv2d(x, filter, padding=1, groups=x.size(1))

# region General Blocks




class Lm_linear_encoder(nn.Module):
    def __init__(self, input_size=136, hidden_size = 136, output_size=136):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        encode = []
        # self.decode = nn.ModuleList()

        repeat_num =  5
        # print('repeat_num', repeat_num)
        dim_in = self.input_size
        dim_out = self.hidden_size
        for _ in range(repeat_num):


            encode += [nn.Linear(dim_in, dim_out)]
            # encode += [nn.Dropout(0.2)]
            # encode += [nn.ReLU(inplace=True)]
            encode += [nn.LeakyReLU(0.2)]

            dim_in = dim_out
        # encode += [nn.Linear(dim_out, dim_out)]
        # encode += [nn.ReLU(inplace=True)]
        encode += [nn.Linear(dim_out, self.output_size)]
        self.main = nn.Sequential(*encode)
    def forward(self, x):

        out = self.main(x)

        return out



class Linear_decoder(nn.Module):
    def __init__(self, input_size=4232, hidden_size = 512, output_size=136):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        decode = []
        # self.decode = nn.ModuleList()

        # repeat_num =  5
        repeat_num =  5
        # print('repeat_num', repeat_num)
        dim_in = self.input_size
        dim_out = self.hidden_size
        
        for _ in range(repeat_num):

            decode += [nn.Linear(dim_in, dim_out)]
            # decode += [nn.Dropout(0.2)]
            # encode += [nn.ReLU(inplace=True)]
            decode += [nn.LeakyReLU(0.2)]
            dim_in = dim_out

        # decode += [nn.Linear(dim_out, dim_out)]
        # decode += [nn.ReLU(inplace=True)]
        decode += [nn.Linear(dim_out, self.output_size)]
        self.main = nn.Sequential(*decode)
    def forward(self, x,y):
        # print(x.shape)
        # print(y.shape)
        input = torch.cat((x, y), dim=1)
        # print(input.shape)
        # assert False

        out = self.main(input)
        out =torch.sigmoid(out)

        return out

class Linear_discriminator(nn.Module):
    def __init__(self, input_size=136, hidden_size = 136, output_size_1=1, output_size_2=150):
    # def __init__(self, input_size=136, hidden_size=512, output_size_1=1, output_size_2=150):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size_1 = output_size_1
        self.output_size_2 = output_size_2
        discriminator = []
        # self.decode = nn.ModuleList()

        repeat_num =  5
        # print('repeat_num', repeat_num)
        dim_in = self.input_size
        dim_out = self.hidden_size
        
        for _ in range(repeat_num):


            discriminator += [nn.Linear(dim_in, dim_out)]
            # discriminator += [nn.Dropout(0.2)]
            discriminator += [nn.LeakyReLU(0.2)]
            # discriminator += [nn.Dropout(0.2)]
            # discriminator += [nn.ReLU()]

            dim_in = dim_out

        output_1 = []
        output_1 += [nn.Linear(dim_out, self.output_size_1)]

        # output_2 = []
        # output_2 += [nn.Linear(dim_out, self.output_size_2)]


        self.main = nn.Sequential(*discriminator)
        self.main_2 = nn.Sequential(*output_1)
        # self.main_3 = nn.Sequential(*output_2)


    def forward(self, x):

        out = self.main(x)

        out_1 = self.main_2(out)
        # out_1 = torch.sigmoid(out_1)
        # out_2 = self.main_3(out)
        # out_2 =torch.sigmoid(out_2)
        

        return out_1


class Linear_discriminator_pair(nn.Module):
    def __init__(self, input_size=136, hidden_size=512, output_size_1=128, output_size_2=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size_1 = output_size_1
        self.output_size_2 = output_size_2
        discriminator_face_1 = []
        discriminator_face_2 = []
        discriminator_pair = []
        # self.decode = nn.ModuleList()

        repeat_num = 2
        # print('repeat_num', repeat_num)
        dim_in = self.input_size
        dim_out = self.hidden_size

        for _ in range(repeat_num):
            discriminator_face_1 += [nn.Linear(dim_in, dim_out)]
            discriminator_face_2 += [nn.Linear(dim_in, dim_out)]
            # discriminator += [nn.Dropout(0.2)]
            discriminator_face_1 += [nn.LeakyReLU(0.2)]
            discriminator_face_2 += [nn.LeakyReLU(0.2)]
            # discriminator += [nn.Dropout(0.2)]
            # discriminator += [nn.ReLU()]

            dim_in = dim_out
        discriminator_face_1 += [nn.Linear(512, output_size_1)]
        discriminator_face_2 += [nn.Linear(512, output_size_1)]

        dim_in = 256
        dim_out = output_size_1
        # print(dim_in, dim_out)
        for _ in range(repeat_num):
            discriminator_pair += [nn.Linear(dim_in, dim_out)]
            discriminator_pair += [nn.LeakyReLU(0.2)]

            dim_in = dim_out
            dim_out = dim_out//2

        discriminator_pair += [nn.Linear(64, self.output_size_2)]
        # output_1 = []
        # discriminator_pair += [nn.Linear(dim_out, self.output_size_2)]

        # output_2 = []
        # output_2 += [nn.Linear(dim_out, self.output_size_2)]

        self.face_1 = nn.Sequential(*discriminator_face_1)
        self.face_2 = nn.Sequential(*discriminator_face_2)
        self.pair_1 = nn.Sequential(*discriminator_pair)
        # self.main_2 = nn.Sequential(*output_1)
        # self.main_3 = nn.Sequential(*output_2)

    def forward(self, x, y):
        out_x = self.face_1(x)
        out_y = self.face_2(y)
        out = torch.cat([out_x, out_y], dim=1)

        out_1 = self.pair_1(out)
        # out_1 = torch.sigmoid(out_1)
        # out_2 = self.main_3(out)
        # out_2 =torch.sigmoid(out_2)

        return out_1




class Linear_classfier(nn.Module):
    def __init__(self, input_size=136, hidden_size=136, output_size_1=1, output_size_2=12606):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size_1 = output_size_1
        self.output_size_2 = output_size_2
        discriminator = []
        # self.decode = nn.ModuleList()

        repeat_num = 5
        # print('repeat_num', repeat_num)
        dim_in = self.input_size
        dim_out = self.hidden_size

        for _ in range(repeat_num):
            discriminator += [nn.Linear(dim_in, dim_out)]
            ##
            # discriminator += [nn.Dropout(0.2)]
            ##
            discriminator += [nn.LeakyReLU(0.2)]
            # discriminator += [nn.Dropout(0.2)]
            # discriminator += [nn.ReLU()]

            dim_in = dim_out

        # output_1 = []
        # output_1 += [nn.Linear(dim_out, self.output_size_1)]

        output_2 = []
        output_2 += [nn.Linear(dim_out, self.output_size_2)]

        self.main = nn.Sequential(*discriminator)
        # self.main_2 = nn.Sequential(*output_1)
        self.main_3 = nn.Sequential(*output_2)

    def forward(self, x):
        out = self.main(x)

        # out_1 = self.main_2(out)
        # out_1 = torch.sigmoid(out_1)
        out_2 = self.main_3(out)
        out_2 = torch.sigmoid(out_2)

        return out_2




def build_model_idsg(args):
    if args.transformer:

        linear_decoder = Linear_decoder()
        lm_linear_encoder = Lm_linear_encoder()
        linear_discriminator = Linear_discriminator()
        if args.dataset == 'vox1':

            # linear_classfier = Linear_classfier(output_size_2 = 1000)
            # linear_classfier = Linear_classfier(output_size_2=12606)
            linear_classfier = Linear_classfier(output_size_2=1454)
        elif args.dataset == 'rafd':

            linear_classfier = Linear_classfier(output_size_2 = 67)


        linear_decoder_ema = copy.deepcopy(linear_decoder)
        lm_linear_encoder_ema = copy.deepcopy(lm_linear_encoder)
        linear_discriminator_ema = copy.deepcopy(linear_discriminator)
        linear_classfier_ema = copy.deepcopy(linear_classfier)



        nets = Munch(linear_decoder=linear_decoder,
                     lm_linear_encoder=lm_linear_encoder,
                     linear_discriminator=linear_discriminator,
                     linear_classfier=linear_classfier)


        nets_ema = Munch(linear_decoder=linear_decoder_ema,
                     lm_linear_encoder=lm_linear_encoder_ema,
                     linear_discriminator=linear_discriminator_ema,
                         linear_classfier=linear_classfier_ema)

    else:
        print('??')
        assert False


    # else:
    #     generator = Generator(args.img_size, args.style_dim, w_hpf=args.w_hpf)
    #     style_encoder = StyleEncoder(args.img_size, args.style_dim, args.num_domains)
    #     discriminator_img = Discriminator_img(args.img_size, args.num_domains)
    #     generator_ema = copy.deepcopy(generator)
    #     style_encoder_ema = copy.deepcopy(style_encoder)
    #
    #
    #     nets = Munch(generator=generator,
    #                  style_encoder=style_encoder,
    #                  discriminator=discriminator_img)
    #     nets_ema = Munch(generator=generator_ema,
    #                      style_encoder=style_encoder_ema)

    # if args.w_hpf > 0:
    #     fan = FAN(fname_pretrained=args.wing_path).eval()
    #     nets.fan = fan
    #     nets_ema.fan = fan

    return nets, nets_ema
