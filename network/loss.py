from network.vgg import vgg_face, VGG_Activations, VGG_Activations_2
from torchvision.models import vgg19

import torch
from torch import nn
from torch.nn import functional as F

# import config


class LossEG(nn.Module):
    def __init__(self, feed_forward=True, gpu=None):
        super(LossEG, self).__init__()

        self.VGG_FACE_AC = VGG_Activations(vgg_face(pretrained=True), [1, 6, 11, 18, 25])


        # self.match_loss = not feed_forward
        self.gpu = gpu
        if gpu is not None:
            self.cuda(gpu)

    def loss_cnt(self, x, x_hat):
        IMG_NET_MEAN = torch.Tensor([0.485, 0.456, 0.406]).reshape([1, 3, 1, 1]).to(x.device)
        IMG_NET_STD = torch.Tensor([0.229, 0.224, 0.225]).reshape([1, 3, 1, 1]).to(x.device)

        x = (x - IMG_NET_MEAN) / IMG_NET_STD
        x_hat = (x_hat - IMG_NET_MEAN) / IMG_NET_STD


        # VGG Face Loss
        vgg_face_x_hat = self.VGG_FACE_AC(x_hat)
        vgg_face_x = self.VGG_FACE_AC(x)

        vgg_face_loss = 0
        for i in range(0, len(vgg_face_x)):
            vgg_face_loss += F.l1_loss(vgg_face_x_hat[i], vgg_face_x[i])

        return vgg_face_loss


    def forward(self, x, x_hat):
        if self.gpu is not None:
            x = x.cuda(self.gpu)
            x_hat = x_hat.cuda(self.gpu)


        cnt = self.loss_cnt(x, x_hat)


        return cnt .reshape(1)

class vgg_feature(nn.Module):
    def __init__(self, feed_forward=True, gpu=None):
        super(vgg_feature, self).__init__()

        self.VGG_FACE_AC = VGG_Activations_2(vgg_face(pretrained=True), [1, 6, 11, 18, 25])


        # self.match_loss = not feed_forward
        self.gpu = gpu
        if gpu is not None:
            self.cuda(gpu)

    def extract_fea(self, x):
        IMG_NET_MEAN = torch.Tensor([0.485, 0.456, 0.406]).reshape([1, 3, 1, 1]).to(x.device)
        IMG_NET_STD = torch.Tensor([0.229, 0.224, 0.225]).reshape([1, 3, 1, 1]).to(x.device)

        x = (x - IMG_NET_MEAN) / IMG_NET_STD
        # x_hat = (x_hat - IMG_NET_MEAN) / IMG_NET_STD


        # VGG Face Loss
        # vgg_face_x_hat = self.VGG_FACE_AC(x_hat)
        vgg_face_x = self.VGG_FACE_AC(x)

        # vgg_face_loss = 0
        # for i in range(0, len(vgg_face_x)):
        #     vgg_face_loss += F.l1_loss(vgg_face_x_hat[i], vgg_face_x[i])

        return vgg_face_x


    def forward(self, x):
        if self.gpu is not None:
            x = x.cuda(self.gpu)
            # x_hat = x_hat.cuda(self.gpu)


        fea = self.extract_fea(x)
        # print(type(fea))
        # print(fea.shape)

        return fea

class LossD(nn.Module):
    def __init__(self, gpu=None):
        super(LossD, self).__init__()
        self.gpu = gpu
        if gpu is not None:
            self.cuda(gpu)

    def forward(self, r_x, r_x_hat):
        if self.gpu is not None:
            r_x = r_x.cuda(self.gpu)
            r_x_hat = r_x_hat.cuda(self.gpu)
        return (F.relu(1 + r_x_hat) + F.relu(1 - r_x)).mean().reshape(1)
