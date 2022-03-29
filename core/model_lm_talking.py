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


class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample=False):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance


class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta

class NoiseLayer(nn.Module):
    """adds noise. noise is per pixel (constant over channels) with per-channel weight"""

    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(channels))
        self.noise = None

    def forward(self, x, noise=None):
        # print(x)
        # print(noise)
        # print(self.noise)
        if noise is None and self.noise is None:
            # print('add noise')
            # print(x.device)
            noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
        elif noise is None:
            print(x.device)
            # here is a little trick: if you get all the noise layers and set each
            # modules .noise attribute, you can have pre-defined noise.
            # Very useful for analysis
            noise = self.noise
        x = x + self.weight.view(1, -1, 1, 1) * noise
        return x


class AdainResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64, w_hpf=0,
                 actv=nn.LeakyReLU(0.2), upsample=False, use_noise=False):
        super().__init__()
        self.w_hpf = w_hpf
        self.actv = actv
        self.upsample = upsample
        self.use_noise = use_noise
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)
        self.dim_in = dim_in
        self.dim_out = dim_out
        # layers = []
        if self.use_noise:
            # layers.append(('noise', NoiseLayer(channels)))
            self.noise = NoiseLayer(dim_in)
            self.noise_2 = NoiseLayer(dim_out)

    def _build_weights(self, dim_in, dim_out, style_dim=64):
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        self.norm1 = AdaIN(style_dim, dim_in)
        self.norm2 = AdaIN(style_dim, dim_out)

        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        # print(self.use_noise)
        if self.use_noise:

            x = self.noise(x)
            x = self.actv(x)
        # print(x)
        x = self.norm1(x, s)
        x = self.actv(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1(x)
        if self.use_noise:
            x = self.noise_2(x)
            x = self.actv(x)

        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        if self.w_hpf == 0:
            out = (out + self._shortcut(x)) / math.sqrt(2)
        return out


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

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.rand(1).normal_(0.0, 0.02))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # B: mini batches, C: channels, W: width, H: height
        B, C, H, W = x.shape
        proj_query = self.query_conv(x).view(B, -1, W * H).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(B, -1, W * H)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(B, -1, W * H)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)

        out = self.gamma * out + x

        return out

class SelfAttention_2(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention_2, self).__init__()

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.rand(1).normal_(0.0, 0.02))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, s):
        # B: mini batches, C: channels, W: width, H: height
        B, C, H, W = x.shape
        proj_query = self.query_conv(x).view(B, -1, W * H).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(B, -1, W * H)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(B, -1, W * H)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)

        out = self.gamma * out + x

        return out

class Generator_2(nn.Module):
    def __init__(self, img_size=256, style_dim=64, max_conv_dim=512, w_hpf=1):
        super().__init__()
        dim_in = 2**14 // img_size
        # print(dim_in)
        self.img_size = img_size
        self.from_rgb = nn.Conv2d(3, dim_in, 3, 1, 1)
        # self.from_LM = nn.Conv2d(1, dim_in, 3, 1, 1)
        self.encode = nn.ModuleList()
        self.decode = nn.ModuleList()
        self.to_rgb = nn.Sequential(
            nn.InstanceNorm2d(dim_in, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim_in, 3, 1, 1, 0))
        self.to_rgb_lm = nn.Sequential(
            nn.InstanceNorm2d(dim_in, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim_in, 4, 1, 1, 0))
        # self.att1 = SelfAttention(256)
        # self.att2 = SelfAttention(128)


        # down/up-sampling blocks
        repeat_num = int(np.log2(img_size)) - 4
        # print('repeat_num', repeat_num)
        if w_hpf > 0:
            repeat_num += 1

        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            self.encode.append(
                ResBlk(dim_in, dim_out, normalize=True, downsample=True))
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_in, style_dim,
                               w_hpf=w_hpf, upsample=True))  # stack-like
            dim_in = dim_out

        self.encode.append(SelfAttention(dim_out))
        self.decode.insert(0,SelfAttention_2(dim_in))

        for _ in range(2):
            dim_out = min(dim_in * 2, max_conv_dim)
            self.encode.append(
                ResBlk(dim_in, dim_out, normalize=True, downsample=True))
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_in, style_dim,
                               w_hpf=w_hpf, upsample=True))  # stack-like
            dim_in = dim_out


        # bottleneck blocks
        for _ in range(5):
            # self.encode.append(
            #     ResBlk(dim_out, dim_out, normalize=True))
            # self.encode.append(
            #     AdainResBlk(dim_out, dim_out, style_dim, w_hpf=w_hpf))
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_out, style_dim, w_hpf=w_hpf))

        if w_hpf > 0:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
            self.hpf = HighPass(w_hpf, device)

    def forward(self, x, s, masks=None):
        x = self.from_rgb(x)
        # x = self.from_LM(x)
        cache = {}
        for block in self.encode:
            if (masks is not None) and (x.size(2) in [32, 64, 128]):
                cache[x.size(2)] = x
            x = block(x)
        # print(x.shape)
        # print(s.shape)
        # print(self.decode)
        for block in self.decode:
            x = block(x, s)
            if (masks is not None) and (x.size(2) in [32, 64, 128]):
                # mask = masks[0] if x.size(2) in [32] else masks[1]
                mask = masks[:, 0:1, :, :]
                mask = F.interpolate(mask, size=x.size(2), mode='bilinear')
                x = x + self.hpf(mask * cache[x.size(2)])

        # return self.to_rgb(x)
        return torch.sigmoid(self.to_rgb(x))


class Generator(nn.Module):
    def __init__(self, img_size=256, style_dim=64, max_conv_dim=512, w_hpf=1, use_noise=False):
        super().__init__()
        dim_in = 2**14 // img_size
        # print(dim_in)
        self.img_size = img_size
        self.use_noise = use_noise
        self.from_rgb = nn.Conv2d(3, dim_in, 3, 1, 1)
        # self.from_LM = nn.Conv2d(1, dim_in, 3, 1, 1)
        self.encode = nn.ModuleList()
        self.decode = nn.ModuleList()
        self.to_rgb = nn.Sequential(
            nn.InstanceNorm2d(dim_in, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim_in, 3, 1, 1, 0))
        self.to_rgb_lm = nn.Sequential(
            nn.InstanceNorm2d(dim_in, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim_in, 4, 1, 1, 0))

        # down/up-sampling blocks
        repeat_num = int(np.log2(img_size)) - 4
        # print('repeat_num', repeat_num)
        if w_hpf > 0:
            repeat_num += 1

        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            self.encode.append(
                ResBlk(dim_in, dim_out, normalize=True, downsample=True))
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_in, style_dim,
                               w_hpf=w_hpf, upsample=True, use_noise=self.use_noise))  # stack-like
            dim_in = dim_out

        # bottleneck blocks
        for _ in range(4):
            # self.encode.append(
            #     ResBlk(dim_out, dim_out, normalize=True))
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_out, style_dim, w_hpf=w_hpf, use_noise=self.use_noise))

        if w_hpf > 0:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
            self.hpf = HighPass(w_hpf, device)

    def forward(self, x, s, masks=None, loss_select='perceptual'):
        x = self.from_rgb(x)
        # x = self.from_LM(x)
        cache = {}
        for block in self.encode:
            if (masks is not None) and (x.size(2) in [32, 64, 128]):
                cache[x.size(2)] = x
            x = block(x)

        for block in self.decode:

            x = block(x, s)

            if (masks is not None) and (x.size(2) in [32, 64, 128]):
                # mask = masks[0] if x.size(2) in [32] else masks[1]
                mask = masks[:, 0:1, :, :]
                mask = F.interpolate(mask, size=x.size(2), mode='bilinear')
                x = x + mask
                # x = x +  self.hpf(mask)
                # x = x + self.hpf(mask * cache[x.size(2)])
        if loss_select == 'perceptual':
            output =torch.sigmoid(self.to_rgb(x))
            
        elif loss_select == 'arcface':
            output = self.to_rgb(x)
        elif loss_select == 'lightcnn':
            output = torch.sigmoid(self.to_rgb(x))
        return output


# class MappingNetwork(nn.Module):
#     def __init__(self, latent_dim=16, style_dim=64, num_domains=2):
#         super().__init__()
#         layers = []
#         layers += [nn.Linear(latent_dim, 512)]
#         layers += [nn.ReLU()]
#         for _ in range(3):
#             layers += [nn.Linear(512, 512)]
#             layers += [nn.ReLU()]
#         self.shared = nn.Sequential(*layers)
#
#         self.unshared = nn.ModuleList()
#         for _ in range(num_domains):
#             self.unshared += [nn.Sequential(nn.Linear(512, 512),
#                                             nn.ReLU(),
#                                             nn.Linear(512, 512),
#                                             nn.ReLU(),
#                                             nn.Linear(512, 512),
#                                             nn.ReLU(),
#                                             nn.Linear(512, style_dim))]
#
#     def forward(self, z, y):
#         h = self.shared(z)
#         out = []
#         for layer in self.unshared:
#             out += [layer(h)]
#         out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)
#         idx = torch.LongTensor(range(y.size(0))).to(y.device)
#         s = out[idx, y]  # (batch, style_dim)
#         return s




class StyleEncoder(nn.Module):
    def __init__(self, img_size=256, style_dim=64, num_domains=2, self_att=False, max_conv_dim=512):
        super().__init__()
        dim_in = 2**14 // img_size
        blocks = []
        blocks += [nn.Conv2d(3, dim_in, 3, 1, 1)]

        self.self_att = self_att
        repeat_num = int(np.log2(img_size)) - 2
        if self.self_att:
            for _ in range(repeat_num-2):
                dim_out = min(dim_in*2, max_conv_dim)
                blocks += [ResBlk(dim_in, dim_out, downsample=True)]
                dim_in = dim_out
            blocks += [SelfAttention(dim_out)]
            for _ in range(2):
                dim_out = min(dim_in*2, max_conv_dim)
                blocks += [ResBlk(dim_in, dim_out, downsample=True)]
                dim_in = dim_out
        else:
            for _ in range(repeat_num):
                dim_out = min(dim_in*2, max_conv_dim)
                blocks += [ResBlk(dim_in, dim_out, downsample=True)]
                dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        self.shared = nn.Sequential(*blocks)

        self.fc = nn.Linear(dim_out, style_dim)

        # self.unshared = nn.ModuleList()
        # for _ in range(num_domains):
        #     self.unshared += [nn.Linear(dim_out, style_dim)]

    def forward(self, x):
        # out = torch.cat((x, y), dim=1)
        h = self.shared(x)
        h = h.view(h.size(0), -1)
        h = self.fc(h)
        # print(h.shape)

        return h # (batch, style_dim)

        # print(h.shape)
        # assert False
        # out = []
        # for layer in self.unshared:
        #     out += [layer(h)]
        # out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)
        # idx = torch.LongTensor(range(y.size(0))).to(y.device)
        # s = out[idx, y]  # (batch, style_dim)
        # return s
class Discriminator_img_pix(nn.Module):
    def __init__(self, img_size=256, num_domains=2, self_att=False, max_conv_dim=512):
        super().__init__()
        dim_in = 2**14 // img_size
        self.self_att = self_att
        self.input = nn.Conv2d(6, dim_in, 3, 1, 1)
        self.blocks = nn.ModuleList()

        # blocks = []
        # blocks += [nn.Conv2d(6, dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(img_size)) - 2
        if self.self_att:
            for _ in range(repeat_num-2):
                dim_out = min(dim_in*2, max_conv_dim)
                self.blocks.append(ResBlk(dim_in, dim_out, downsample=True))
                dim_in = dim_out
            self.blocks.append(SelfAttention(dim_out))
            for _ in range(2):
                dim_out = min(dim_in*2, max_conv_dim)
                self.blocks.append(ResBlk(dim_in, dim_out, downsample=True))
                dim_in = dim_out
        else:
            for _ in range(repeat_num):
                dim_out = min(dim_in*2, max_conv_dim)
                self.blocks.append(ResBlk(dim_in, dim_out, downsample=True))
                dim_in = dim_out

        self.output = nn.Sequential(nn.LeakyReLU(0.2), nn.Conv2d(dim_out, dim_out, 4, 1, 0), nn.LeakyReLU(0.2),  nn.Conv2d(dim_out, 1, 1, 1, 0))
        # blocks += [nn.LeakyReLU(0.2)]
        # blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
        # blocks += [nn.LeakyReLU(0.2)]
        # blocks += [nn.Conv2d(dim_out, 1, 1, 1, 0)]
        # self.main = nn.Sequential(*blocks)
        # print(dim_out)
        # self.fc = nn.Linear(dim_out, style_dim)

    def forward(self, x, y):
        out = torch.cat((x, y), dim=1)
        out = self.input(out)
        fea = []
        for num,block in enumerate(self.blocks):
            # print(num)
            out = block(out)
            fea.append(out)
        out =  self.output(out)
        out = out.view(out.size(0), -1)  # (batch, num_domains)
        # print(out.shape)
        # assert False
        #
        # idx = torch.LongTensor(range(y.size(0))).to(y.device)
        # out = out[idx, y]  # (batch)
        return fea, out

class Discriminator_img2_pix(nn.Module):
    def __init__(self, img_size=256, num_domains=2, self_att=False, max_conv_dim=512):
        super().__init__()
        dim_in = 2**14 // img_size
        self.self_att = self_att
        self.input = nn.Conv2d(6, dim_in, 3, 1, 1)
        self.blocks = nn.ModuleList()

        # blocks = []
        # blocks += [nn.Conv2d(6, dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(img_size)) - 2
        if self.self_att:
            for _ in range(repeat_num-2):
                dim_out = min(dim_in*2, max_conv_dim)
                self.blocks.append(ResBlk(dim_in, dim_out, downsample=True))
                dim_in = dim_out
            self.blocks.append(SelfAttention(dim_out))
            for _ in range(2):
                dim_out = min(dim_in*2, max_conv_dim)
                self.blocks.append(ResBlk(dim_in, dim_out, downsample=True))
                dim_in = dim_out
        else:
            for _ in range(repeat_num):
                dim_out = min(dim_in*2, max_conv_dim)
                self.blocks.append(ResBlk(dim_in, dim_out, downsample=True))
                dim_in = dim_out

        self.output = nn.Sequential(nn.LeakyReLU(0.2), nn.Conv2d(dim_out, dim_out, 2, 1, 0), nn.LeakyReLU(0.2),  nn.Conv2d(dim_out, 1, 1, 1, 0))
        # blocks += [nn.LeakyReLU(0.2)]
        # blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
        # blocks += [nn.LeakyReLU(0.2)]
        # blocks += [nn.Conv2d(dim_out, 1, 1, 1, 0)]
        # self.main = nn.Sequential(*blocks)
        # print(dim_out)
        # self.fc = nn.Linear(dim_out, style_dim)

    def forward(self, x, y):
        x = F.avg_pool2d(x, 2)
        y = F.avg_pool2d(y, 2)
        out = torch.cat((x, y), dim=1)
        out = self.input(out)
        fea = []
        for num,block in enumerate(self.blocks):

            out = block(out)
            # print(num.shape)
            fea.append(out)
        out =  self.output(out)
        out = out.view(out.size(0), -1)  # (batch, num_domains)
        # print(out.shape)
        # assert False
        #
        # idx = torch.LongTensor(range(y.size(0))).to(y.device)
        # out = out[idx, y]  # (batch)
        return fea, out

class Discriminator_img(nn.Module):
    def __init__(self, img_size=256, num_domains=2, self_att=False, max_conv_dim=512):
        super().__init__()
        dim_in = 2**14 // img_size
        blocks = []
        blocks += [nn.Conv2d(6, dim_in, 3, 1, 1)]
        self.self_att = self_att
        repeat_num = int(np.log2(img_size)) - 2
        if self.self_att:
            for _ in range(repeat_num-2):
                dim_out = min(dim_in*2, max_conv_dim)
                blocks += [ResBlk(dim_in, dim_out, downsample=True)]
                dim_in = dim_out
            blocks += [SelfAttention(dim_out)]
            for _ in range(2):
                dim_out = min(dim_in*2, max_conv_dim)
                blocks += [ResBlk(dim_in, dim_out, downsample=True)]
                dim_in = dim_out
        else:
            for _ in range(repeat_num):
                dim_out = min(dim_in*2, max_conv_dim)
                blocks += [ResBlk(dim_in, dim_out, downsample=True)]
                dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
        # print()
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, 1, 1, 1, 0)]
        self.main = nn.Sequential(*blocks)
        # print(dim_out)
        # self.main2 = nn.LeakyReLU(0.2)
        # self.main3 = nn.Conv2d(dim_out, dim_out, 4, 1, 0)
        # self.main4 = nn.LeakyReLU(0.2)
        # self.main5 = nn.Conv2d(dim_out, 1, 1, 1, 0)

    def forward(self, x, y):
        out = torch.cat((x, y), dim=1)
        out = self.main(out)
        # print(out.shape)
        # out = self.main2(out)
        # print(out.shape)
        # out = self.main3(out)
        # print(out.shape)
        # out = self.main4(out)
        # print(out.shape)
        # out = self.main5(out)
        # print(out.shape)
        # assert False
        out = out.view(out.size(0), -1)  # (batch, num_domains)
        # print(out.shape)
        #
        # idx = torch.LongTensor(range(y.size(0))).to(y.device)
        # out = out[idx, y]  # (batch)
        return out

class Discriminator_img2(nn.Module):
    def __init__(self, img_size=256, num_domains=2, self_att=False, max_conv_dim=512):
        super().__init__()
        dim_in = 2**14 // img_size
        blocks = []
        blocks += [nn.Conv2d(6, dim_in, 3, 1, 1)]
        self.self_att = self_att
        repeat_num = int(np.log2(img_size)) - 2
        if self.self_att:
            for _ in range(repeat_num-2):
                dim_out = min(dim_in*2, max_conv_dim)
                blocks += [ResBlk(dim_in, dim_out, downsample=True)]
                dim_in = dim_out
            blocks += [SelfAttention(dim_out)]
            for _ in range(2):
                dim_out = min(dim_in*2, max_conv_dim)
                blocks += [ResBlk(dim_in, dim_out, downsample=True)]
                dim_in = dim_out
        else:
            for _ in range(repeat_num):
                dim_out = min(dim_in*2, max_conv_dim)
                blocks += [ResBlk(dim_in, dim_out, downsample=True)]
                dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 2, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, 1, 1, 1, 0)]
        self.main = nn.Sequential(*blocks)
        # print(dim_out)
        # self.fc = nn.Linear(dim_out, style_dim)
        # self.main = nn.Sequential(*blocks)
        # print(dim_out)
        # self.main2 = nn.LeakyReLU(0.2)
        # self.main3 = nn.Conv2d(dim_out, dim_out, 2, 1, 0)
        # self.main4 = nn.LeakyReLU(0.2)
        # self.main5 = nn.Conv2d(dim_out, 1, 1, 1, 0)

    def forward(self, x, y):
        x = F.avg_pool2d(x, 2)
        y = F.avg_pool2d(y, 2)
        out = torch.cat((x, y), dim=1)
        out = self.main(out)
        # print(out.shape)
        # out = self.main2(out)
        # print(out.shape)
        # out = self.main3(out)
        # print(out.shape)
        # out = self.main4(out)
        # print(out.shape)
        # out = self.main5(out)
        # print(out.shape)
        # assert False
        out = out.view(out.size(0), -1)  # (batch, num_domains)

        # print(out.shape)
        #
        # idx = torch.LongTensor(range(y.size(0))).to(y.device)
        # out = out[idx, y]  # (batch)
        return out

class MLP(nn.Module):
    def __init__(self, input_dim=4096, style_dim=64, num_domains=2, max_conv_dim=512):
        super().__init__()
        # dim_in = 2**14 // img_size
        blocks = []
        # blocks += [nn.Conv2d(3, dim_in, 3, 1, 1)]
        #
        # repeat_num = int(np.log2(img_size)) - 2
        # for _ in range(repeat_num-2):
        #     dim_out = min(dim_in*2, max_conv_dim)
        #     blocks += [ResBlk(dim_in, dim_out, downsample=True)]
        #     dim_in = dim_out
        # blocks += [SelfAttention(dim_out)]
        # for _ in range(2):
        #     dim_out = min(dim_in*2, max_conv_dim)
        #     blocks += [ResBlk(dim_in, dim_out, downsample=True)]
        #     dim_in = dim_out

        # blocks += [nn.LeakyReLU(0.2)]
        # blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
        # blocks += [nn.LeakyReLU(0.2)]
        # blocks += [nn.Linear(input_dim, input_dim)]
        blocks += [nn.Linear(input_dim, style_dim)]
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Linear(style_dim, style_dim)]
        # blocks += [nn.Linear(style_dim, style_dim)]

        self.mlp = nn.Sequential(*blocks)

        # self.fc = nn.Linear(dim_out, style_dim)

        # self.unshared = nn.ModuleList()
        # for _ in range(num_domains):
        #     self.unshared += [nn.Linear(dim_out, style_dim)]
        # print(self.mlp)

    def forward(self, x):
        # print(x.shape)
        # out = torch.cat((x, y), dim=1)
        # h = self.shared(x)
        # x = x.view(x.size(0), -1)
        h = self.mlp(x)
        # print(h.shape)

        return h # (batch, style_dim)

        # print(h.shape)
        # assert False
        # out = []
        # for layer in self.unshared:
        #     out += [layer(h)]
        # out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)
        # idx = torch.LongTensor(range(y.size(0))).to(y.device)
        # s = out[idx, y]  # (batch, style_dim)
        # return s

def build_model(args):
    if args.id_embed:
        generator = Generator(args.img_size, args.style_dim, w_hpf=args.w_hpf, use_noise=args.use_noise)
        # mapping_network = MappingNetwork(args.latent_dim, args.style_dim, args.num_domains)
        mlp = MLP(args.id_embed_dim, args.style_dim, args.num_domains)
        discriminator_img = Discriminator_img(args.img_size, args.num_domains)
        generator_ema = copy.deepcopy(generator)
        mlp_ema = copy.deepcopy(mlp)
        # mapping_network_ema = copy.deepcopy(mapping_network)
        # style_encoder_ema = copy.deepcopy(style_encoder)

        # nets = Munch(generator=generator,
        #              mapping_network=mapping_network,
        #              style_encoder=style_encoder,
        #              discriminator=discriminator)
        # nets_ema = Munch(generator=generator_ema,
        #                  mapping_network=mapping_network_ema,
        #                  style_encoder=style_encoder_ema)
        nets = Munch(generator=generator, mlp=mlp,
                     discriminator=discriminator_img)
        nets_ema = Munch(generator=generator_ema, mlp=mlp_ema)

        if args.w_hpf > 0:
            fan = FAN(fname_pretrained=args.wing_path).eval()
            nets.fan = fan
            nets_ema.fan = fan

        return nets, nets_ema

    else:
        if args.pix2pix:
            generator = Generator(args.img_size, args.style_dim, w_hpf=args.w_hpf, use_noise=args.use_noise)
            # mapping_network = MappingNetwork(args.latent_dim, args.style_dim, args.num_domains)
            style_encoder = StyleEncoder(args.img_size, args.style_dim, args.num_domains, args.self_att)

            discriminator_img = Discriminator_img_pix(args.img_size, args.num_domains, args.self_att)
            # print(args.img_size)
            # print(args.img_size // 2)
            discriminator_img2 = Discriminator_img2_pix(args.img_size, args.num_domains, args.self_att)
            generator_ema = copy.deepcopy(generator)
            # mapping_network_ema = copy.deepcopy(mapping_network)
            style_encoder_ema = copy.deepcopy(style_encoder)

            # nets = Munch(generator=generator,
            #              mapping_network=mapping_network,
            #              style_encoder=style_encoder,
            #              discriminator=discriminator)
            # nets_ema = Munch(generator=generator_ema,
            #                  mapping_network=mapping_network_ema,
            #                  style_encoder=style_encoder_ema)
            nets = Munch(generator=generator,
                         style_encoder=style_encoder,
                         discriminator=discriminator_img,discriminator2=discriminator_img2)
            nets_ema = Munch(generator=generator_ema,
                             style_encoder=style_encoder_ema)
        else:
            generator = Generator(args.img_size, args.style_dim, w_hpf=args.w_hpf, use_noise=args.use_noise)
            # mapping_network = MappingNetwork(args.latent_dim, args.style_dim, args.num_domains)
            style_encoder = StyleEncoder(args.img_size, args.style_dim, args.num_domains, args.self_att)

            discriminator_img = Discriminator_img(args.img_size, args.num_domains, args.self_att)
            generator_ema = copy.deepcopy(generator)
            # mapping_network_ema = copy.deepcopy(mapping_network)
            style_encoder_ema = copy.deepcopy(style_encoder)

            # nets = Munch(generator=generator,
            #              mapping_network=mapping_network,
            #              style_encoder=style_encoder,
            #              discriminator=discriminator)
            # nets_ema = Munch(generator=generator_ema,
            #                  mapping_network=mapping_network_ema,
            #                  style_encoder=style_encoder_ema)
            nets = Munch(generator=generator,
                         style_encoder=style_encoder,
                         discriminator=discriminator_img)
            nets_ema = Munch(generator=generator_ema,
                             style_encoder=style_encoder_ema)

        # if args.w_hpf > 0:
        #     fan = FAN(fname_pretrained=args.wing_path).eval()
        #     nets.fan = fan
        #     nets_ema.fan = fan

        return nets, nets_ema
