import face_alignment
from skimage import io
import torch
import timeit
import cv2
import numpy as np
import os
from skimage import transform as trans
from core.model_lm_talking import build_model
from core.checkpoint import CheckpointIO
from core_IDSG.model_lm_talking_tran import build_model_idsg
import network as network2
import torch
import torch.nn as nn
import argparse
from os.path import join as ospj
from torchvision import transforms
import torchvision.utils as vutils

def toTensor(img):
    assert type(img) == np.ndarray,'the img type is {}, but ndarry expected'.format(type(img))
    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except:
        img=img
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img.float().div(255).unsqueeze(0)

def tensor_to_np(tensor):
    img = tensor.mul(255).byte()
    img = img.cpu().numpy().squeeze(0).transpose((1, 2, 0))
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

def show_map(landmark):
    N, C= landmark.size()
    img = np.ones((N,3,256, 256))
    img =torch.from_numpy(img)
    for i in range(0, N):
        img_3 = np.zeros((256, 256, 3))
        line_color = (255, 255, 255)
        line_width = 2
        lm_x = []
        lm_y = []
        for num in range(68):
            lm_x.append(landmark[i, 2*num]*256)
            lm_y.append(landmark[i, 2*num+1]*256)

        diff_1 = int(float(lm_x[42])) - int(float(lm_x[36]))
        diff_2 = int(float(lm_x[45])) - int(float(lm_x[39]))

        if diff_1<=30 and int(float(lm_x[30])) > int(float(lm_x[42])):

            for n in range(0, 12):
                cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
                         (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
            for n in range(17, 21):
                cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
                         (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
            # for n in range(22, 26):
            #     cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
            #              (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
            for n in range(27, 30):
                cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
                         (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
            for n in range(31, 33):
                cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
                         (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
            for n in range(36, 41):
                cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
                         (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
            cv2.line(img_3, (int(float(lm_x[36])), int(float(lm_y[36]))),
                     (int(float(lm_x[41])), int(float(lm_y[41]))), line_color, line_width)
            # for n in range(42, 47):
            #     cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
            #              (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
            # cv2.line(img_3, (int(float(lm_x[42])), int(float(lm_y[42]))),
            #          (int(float(lm_x[47])), int(float(lm_y[47]))), line_color, line_width)
            for n in range(48, 51):
                cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
                         (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
            cv2.line(img_3, (int(float(lm_x[51])), int(float(lm_y[51]))),
                     (int(float(lm_x[62])), int(float(lm_y[62]))), line_color, line_width)
            for n in range(60, 62):
                cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
                         (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
            cv2.line(img_3, (int(float(lm_x[60])), int(float(lm_y[60]))),
                     (int(float(lm_x[67])), int(float(lm_y[67]))), line_color, line_width)
            cv2.line(img_3, (int(float(lm_x[66])), int(float(lm_y[66]))),
                     (int(float(lm_x[67])), int(float(lm_y[67]))), line_color, line_width)
            cv2.line(img_3, (int(float(lm_x[57])), int(float(lm_y[57]))),
                     (int(float(lm_x[66])), int(float(lm_y[66]))), line_color, line_width)

            for n in range(57, 59):
                cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
                         (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
            cv2.line(img_3, (int(float(lm_x[48])), int(float(lm_y[48]))),
                     (int(float(lm_x[59])), int(float(lm_y[59]))), line_color, line_width)

        elif diff_2 <= 30 and int(float(lm_x[30])) < int(float(lm_x[42])):

            for n in range(4, 16):
                cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
                         (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
            # for n in range(17, 21):
            #     cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
            #              (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
            for n in range(22, 26):
                cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
                         (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
            for n in range(27, 30):
                cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
                         (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
            for n in range(33, 35):
                cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
                         (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
            # for n in range(36, 41):
            #     cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
            #              (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
            # cv2.line(img_3, (int(float(lm_x[36])), int(float(lm_y[36]))),
            #          (int(float(lm_x[41])), int(float(lm_y[41]))), line_color, line_width)
            for n in range(42, 47):
                cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
                         (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
            cv2.line(img_3, (int(float(lm_x[42])), int(float(lm_y[42]))),
                     (int(float(lm_x[47])), int(float(lm_y[47]))), line_color, line_width)
            for n in range(51, 57):
                cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
                         (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
            cv2.line(img_3, (int(float(lm_x[51])), int(float(lm_y[51]))),
                     (int(float(lm_x[62])), int(float(lm_y[62]))), line_color, line_width)
            for n in range(62, 66):
                cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
                         (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
            cv2.line(img_3, (int(float(lm_x[66])), int(float(lm_y[66]))),
                     (int(float(lm_x[57])), int(float(lm_y[57]))), line_color, line_width)



        else:


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

        tensor = transforms.ToTensor()(img_3)

        img[i,:,:,:] = tensor
        # print(img.shape)
        # assert False
    img = img.type(torch.cuda.FloatTensor)


    return img

def denormalize(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)


def save_image(x, ncol, filename, loss = 'perceptual'):
    if loss == 'arcface':
        x = denormalize(x)
    vutils.save_image(x.cpu(), filename, nrow=ncol, padding=0)

class Solver_2(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.vgg_encode = network2.vgg_feature(False, 0)

        _, self.nets_ema = build_model_idsg(args)

        for name, module in self.nets_ema.items():
            setattr(self, name + '_ema', module)


        self.ckptios = [CheckpointIO(ospj(args.checkpoint_dir, '{}_nets_ema.ckpt'.format(args.resume_iter)), **self.nets_ema)]

        self.to(self.device)

        self._load_checkpoint(args.resume_iter)

    def _load_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.load(step)

    @torch.no_grad()
    def sample(self, src_rgb, ref_lm):

        src_rgb = src_rgb.to(self.device)

        ref_lm = ref_lm.to(self.device)

        args = self.args

        os.makedirs(args.result_dir, exist_ok=True)


        fea_id_1 = self.vgg_encode(src_rgb)

        fea_lm_1 = self.nets_ema.lm_linear_encoder(ref_lm)
        fake_1 = self.nets_ema.linear_decoder(fea_lm_1, fea_id_1)

        fake_1_eye = turn_eye(fake_1, ref_lm)

        fake_lm = show_map(fake_1_eye)
        lm_fake = tensor_to_np(fake_lm[0].unsqueeze(0))
        lm_fake_2 = tensor_to_np(fake_lm[1].unsqueeze(0))
        lm_fake_3 = tensor_to_np(fake_lm[2].unsqueeze(0))
        lm_fake_4 = tensor_to_np(fake_lm[3].unsqueeze(0))


        return lm_fake, lm_fake_2, lm_fake_3, lm_fake_4

class Solver(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        _, self.nets_ema = build_model(args)

        for name, module in self.nets_ema.items():
            setattr(self, name + '_ema', module)


        self.ckptios = [CheckpointIO(ospj(args.checkpoint_dir, '{}_nets_ema.ckpt'.format(args.resume_iter)), **self.nets_ema)]

        self.to(self.device)


        self._load_checkpoint(args.resume_iter)

    def _load_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.load(step)

    @torch.no_grad()
    def sample(self, src, ref):

        ref = ref.to(self.device)
        args = self.args

        if args.masks:
            masks = ref
        else:
            masks = None
        # print(ref.shape)
        # print(src.shape)
        x_fake = self.nets_ema.generator(ref, src, masks=masks, loss_select=args.loss)

        img_fake =tensor_to_np(x_fake[0].unsqueeze(0))
        img_fake_2 = tensor_to_np(x_fake[1].unsqueeze(0))
        img_fake_3 = tensor_to_np(x_fake[2].unsqueeze(0))
        img_fake_4 = tensor_to_np(x_fake[3].unsqueeze(0))


        return img_fake, img_fake_2, img_fake_3, img_fake_4

    @torch.no_grad()
    def extract(self, src):

        src = src.to(self.device)

        s_ref = self.nets_ema.style_encoder(src)

        return s_ref




print("Loading the RFG Model......")


parser = argparse.ArgumentParser()

# model arguments
parser.add_argument('--img_size', type=int, default=256,
                    help='Image resolution')
parser.add_argument('--num_domains', type=int, default=2,
                    help='Number of domains')
parser.add_argument('--latent_dim', type=int, default=16,
                    help='Latent vector dimension')
parser.add_argument('--hidden_dim', type=int, default=512,
                    help='Hidden dimension of mapping network')
parser.add_argument('--style_dim', type=int, default=128,
                    help='Style code dimension')

parser.add_argument('--id_embed_dim', type=int, default=4096,
                    help='ID code dimension by VGGFace')
# loss select
parser.add_argument('--loss', type=str, default='perceptual',
                    help='the type of loss. [arcface| perceptual|lightcnn]')
# dataset select
parser.add_argument('--dataset', type=str, default='vox1',
                    help='the type of loss. [mpie| 300vw | vox1 | rafd]')

parser.add_argument('--lambda_pixel', type=float, default=1,
                    help='Weight for pixel loss')

parser.add_argument('--multi', action='store_true', default=False,
                    help='multi input')

parser.add_argument('--id_embed', action='store_true', default=False,
                    help='multi input')

parser.add_argument('--use_noise', action='store_true', default=False,
                    help='multi input')

parser.add_argument('--l2', action='store_true', default=False,
                    help='use l2 loss')

parser.add_argument('--fea_match', action='store_true', default=False,
                    help='use feature match loss')

parser.add_argument('--lambda_fea_match', type=float, default=1,
                    help='Weight for feature match loss')

parser.add_argument('--id_cyc', action='store_true', default=False,
                    help='use id consistency loss')

parser.add_argument('--masks', action='store_true', default=False,
                    help='use mask injection')

parser.add_argument('--self_att', action='store_true', default=False,
                    help='use self-attention')

parser.add_argument('--pix2pix', action='store_true', default=False,
                    help='use pix2pix loss')

# weight for objective functions

parser.add_argument('--lambda_fm', type=float, default=1,
                    help='Weight for feature match loss')

parser.add_argument('--lambda_id_cyc', type=float, default=1,
                    help='Weight for id cycle consistency loss')

parser.add_argument('--lambda_per', type=float, default=1,
                    help='Weight for perceptual loss')
parser.add_argument('--lambda_con', type=float, default=1,
                    help='Weight for content loss')
parser.add_argument('--lambda_id', type=float, default=1,
                    help='Weight for id loss')
parser.add_argument('--lambda_reg', type=float, default=1,
                    help='Weight for R1 regularization')
parser.add_argument('--lambda_cyc', type=float, default=1,
                    help='Weight for cyclic consistency loss')
parser.add_argument('--lambda_sty', type=float, default=1,
                    help='Weight for style reconstruction loss')
parser.add_argument('--lambda_ds', type=float, default=1,
                    help='Weight for diversity sensitive loss')
parser.add_argument('--ds_iter', type=int, default=100000,
                    help='Number of iterations to optimize diversity sensitive loss')
parser.add_argument('--w_hpf', type=float, default=0,
                    help='weight for high-pass filtering')

# training arguments
parser.add_argument('--randcrop_prob', type=float, default=0.5,
                    help='Probabilty of using random-resized cropping')
parser.add_argument('--total_iters', type=int, default=260000,
                    help='Number of total iterations')
# 260000
# 440000
parser.add_argument('--resume_iter', type=int, default=501500,
                    help='Iterations to resume training/testing')
parser.add_argument('--batch_size', type=int, default=4,
                    help='Batch size for training')
parser.add_argument('--val_batch_size', type=int, default=15,
                    help='Batch size for validation')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='Learning rate for D, E and G')
parser.add_argument('--f_lr', type=float, default=1e-6,
                    help='Learning rate for F')
parser.add_argument('--beta1', type=float, default=0.0,
                    help='Decay rate for 1st moment of Adam')
parser.add_argument('--beta2', type=float, default=0.99,
                    help='Decay rate for 2nd moment of Adam')
parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='Weight decay for optimizer')
parser.add_argument('--num_outs_per_domain', type=int, default=1,
                    help='Number of generated images per domain during sampling')

# misc
parser.add_argument('--mode', type=str,
                    choices=['train', 'sample', 'eval', 'align'],
                    help='This argument is used in solver')
parser.add_argument('--num_workers', type=int, default=0,
                    help='Number of workers used in DataLoader')
parser.add_argument('--seed', type=int, default=777,
                    help='Seed for random number generator')

# directory for training
# parser.add_argument('--train_img_dir', type=str, default='train_list_300VW.txt',
#                     help='Directory containing training images')
# parser.add_argument('--val_img_dir', type=str, default='test_list_300VW.txt',
#                     help='Directory containing validation images')

parser.add_argument('--sample_dir', type=str, default='expr/samples',
                    help='Directory for saving generated images')
parser.add_argument('--checkpoint_dir', type=str, default='./expr/checkpoints',
                    help='Directory for saving network checkpoints')

# directory for calculating metrics
parser.add_argument('--eval_dir', type=str, default='expr/eval',
                    help='Directory for saving metrics, i.e., FID and LPIPS')

# directory for testing
parser.add_argument('--result_dir', type=str, default='expr/results',
                    help='Directory for saving generated images and videos')
parser.add_argument('--src_dir', type=str, default='test_list_300VW.txt',
                    help='Directory containing input source images')
parser.add_argument('--ref_dir', type=str, default='assets/representative/celeba_hq/ref',
                    help='Directory containing input reference images')
parser.add_argument('--inp_dir', type=str, default='assets/representative/custom/female',
                    help='input directory when aligning faces')
parser.add_argument('--out_dir', type=str, default='assets/representative/celeba_hq/src/female',
                    help='output directory when aligning faces')

# face alignment
parser.add_argument('--wing_path', type=str, default='expr/checkpoints/wing.ckpt')
parser.add_argument('--lm_path', type=str, default='expr/checkpoints/celeba_lm_mean.npz')

# step size
parser.add_argument('--print_every', type=int, default=100)
parser.add_argument('--sample_every', type=int, default=100)
parser.add_argument('--save_every', type=int, default=1000)
parser.add_argument('--eval_every', type=int, default=500000)

args = parser.parse_args()


solver = Solver(args)


print("Loading the IDSG Model......")

parser = argparse.ArgumentParser()

# model arguments
parser.add_argument('--img_size', type=int, default=256,
                    help='Image resolution')
parser.add_argument('--num_domains', type=int, default=2,
                    help='Number of domains')
parser.add_argument('--latent_dim', type=int, default=16,
                    help='Latent vector dimension')
parser.add_argument('--hidden_dim', type=int, default=512,
                    help='Hidden dimension of mapping network')
parser.add_argument('--style_dim', type=int, default=64,
                    help='Style code dimension')
# loss select
parser.add_argument('--loss', type=str, default='perceptual',
                    help='the type of loss. [perceptual]')
# dataset select
parser.add_argument('--dataset', type=str, default='vox1',
                    help='the type of loss. [rafd | vox1]')
# landmark_transformer
parser.add_argument('--transformer', action='store_true', default=True,
                    help='Use landmark_transformer')

parser.add_argument('--vgg_encode', action='store_true', default=True,
                    help='Use landmark_transformer')

# weight for objective functions
parser.add_argument('--lambda_cyc2', type=float, default=1,
                    help='Weight for cyclic consistency loss')
parser.add_argument('--lambda_eye_dis', type=float, default=1,
                    help='Weight for eye distance loss')
parser.add_argument('--lambda_cyc_eye', type=float, default=1,
                    help='Weight for cycle local loss')
parser.add_argument('--lambda_cyc_local', type=float, default=1,
                    help='Weight for cycle local loss')
parser.add_argument('--lambda_tp', type=float, default=1,
                    help='Weight for tp loss')
parser.add_argument('--lambda_cls', type=float, default=1,
                    help='Weight for au loss')
parser.add_argument('--lambda_d', type=float, default=1,
                    help='Weight for au loss')
parser.add_argument('--lambda_pe', type=float, default=1,
                    help='Weight for au loss')
parser.add_argument('--lambda_lm', type=float, default=1,
                    help='Weight for au loss')
parser.add_argument('--lambda_au_aus', type=float, default=1,
                    help='Weight for au loss')
parser.add_argument('--lambda_au_pose', type=float, default=1,
                    help='Weight for au loss')
parser.add_argument('--lambda_per', type=float, default=1,
                    help='Weight for perceptual loss')
parser.add_argument('--lambda_con', type=float, default=1,
                    help='Weight for content loss')
parser.add_argument('--lambda_id', type=float, default=1,
                    help='Weight for id loss')
parser.add_argument('--lambda_reg', type=float, default=1,
                    help='Weight for R1 regularization')
parser.add_argument('--lambda_cyc', type=float, default=1,
                    help='Weight for cyclic consistency loss')
parser.add_argument('--lambda_sty', type=float, default=1,
                    help='Weight for style reconstruction loss')
parser.add_argument('--lambda_ds', type=float, default=1,
                    help='Weight for diversity sensitive loss')
parser.add_argument('--ds_iter', type=int, default=100000,
                    help='Number of iterations to optimize diversity sensitive loss')
parser.add_argument('--w_hpf', type=float, default=1,
                    help='weight for high-pass filtering')

# training arguments
parser.add_argument('--randcrop_prob', type=float, default=0.5,
                    help='Probabilty of using random-resized cropping')
parser.add_argument('--total_iters', type=int, default=100010,
                    help='Number of total iterations')

#40000
parser.add_argument('--resume_iter', type=int, default=30000,
                    help='Iterations to resume training/testing')
parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch size for training')
parser.add_argument('--val_batch_size', type=int, default=48,
                    help='Batch size for validation')
parser.add_argument('--lr', type=float, default=2e-5,
                    help='Learning rate for D, E and G')
parser.add_argument('--lr2', type=float, default=2e-5,
                    help='Learning rate for C')
parser.add_argument('--f_lr', type=float, default=1e-6,
                    help='Learning rate for F')
parser.add_argument('--beta1', type=float, default=0.99,
                    help='Decay rate for 1st moment of Adam')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='Decay rate for 2nd moment of Adam')
parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='Weight decay for optimizer')
parser.add_argument('--num_outs_per_domain', type=int, default=1,
                    help='Number of generated images per domain during sampling')

# misc
parser.add_argument('--mode', type=str,
                    choices=['train', 'sample', 'eval', 'align'],
                    help='This argument is used in solver')
parser.add_argument('--num_workers', type=int, default=0,
                    help='Number of workers used in DataLoader')
parser.add_argument('--seed', type=int, default=777,
                    help='Seed for random number generator')

# directory for training
# parser.add_argument('--train_img_dir', type=str, default='train_list_300VW.txt',
#                     help='Directory containing training images')
# parser.add_argument('--val_img_dir', type=str, default='test_list_300VW.txt',
#                     help='Directory containing validation images')

parser.add_argument('--sample_dir', type=str, default='expr/samples',
                    help='Directory for saving generated images')
parser.add_argument('--checkpoint_dir', type=str, default='./expr_lm/checkpoints',
                    help='Directory for saving network checkpoints')

# directory for calculating metrics
parser.add_argument('--eval_dir', type=str, default='expr/eval',
                    help='Directory for saving metrics, i.e., FID and LPIPS')

# directory for testing
parser.add_argument('--result_dir', type=str, default='./expr_lm/results',
                    help='Directory for saving generated images and videos')
parser.add_argument('--src_dir', type=str, default='test_list_300VW.txt',
                    help='Directory containing input source images')
parser.add_argument('--ref_dir', type=str, default='assets/representative/celeba_hq/ref',
                    help='Directory containing input reference images')
parser.add_argument('--inp_dir', type=str, default='assets/representative/custom/female',
                    help='input directory when aligning faces')
parser.add_argument('--out_dir', type=str, default='assets/representative/celeba_hq/src/female',
                    help='output directory when aligning faces')

# face alignment
parser.add_argument('--wing_path', type=str, default='expr/checkpoints/wing.ckpt')
parser.add_argument('--lm_path', type=str, default='expr/checkpoints/celeba_lm_mean.npz')

# step size
parser.add_argument('--print_every', type=int, default=100)
parser.add_argument('--sample_every', type=int, default=2500)
parser.add_argument('--save_every', type=int, default=10000)
parser.add_argument('--eval_every', type=int, default=100000)
parser.add_argument('--decay_every', type=int, default=10000)

args = parser.parse_args()

solver_2 = Solver_2(args)

print("Loading the FAN Model......")
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True, device='cuda')


print("Loading the Camera......")
print(" ")
print("If you want to exit,please press (ESC) !!")
cap = cv2.VideoCapture(0)


width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)


cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 5.0, (1024, 1024))


def drawshape(landmarks):
    img = np.zeros((256, 256, 3))
    line_color = (255, 255, 255)
    line_width = 2

    diff = int(float(landmarks[42][0])) - int(float(landmarks[36][0]))

    if diff<=30 and int(float(landmarks[30][0])) > int(float(landmarks[42][0])):
        # print('right')

        for n in range(0, 12):
            cv2.line(img, (int(float(landmarks[n][0])), int(float(landmarks[n][1]))),
                     (int(float(landmarks[n + 1][0])), int(float(landmarks[n + 1][1]))), line_color, line_width)
        for n in range(17, 21):
            cv2.line(img, (int(float(landmarks[n][0])), int(float(landmarks[n][1]))),
                     (int(float(landmarks[n + 1][0])), int(float(landmarks[n + 1][1]))), line_color, line_width)
        # for n in range(22, 26):
        #     cv2.line(img, (int(float(landmarks[n][0])), int(float(landmarks[n][1]))),
        #              (int(float(landmarks[n + 1][0])), int(float(landmarks[n + 1][1]))), line_color, line_width)
        for n in range(27, 30):
            cv2.line(img, (int(float(landmarks[n][0])), int(float(landmarks[n][1]))),
                     (int(float(landmarks[n + 1][0])), int(float(landmarks[n + 1][1]))), line_color, line_width)
        for n in range(31, 33):
            cv2.line(img, (int(float(landmarks[n][0])), int(float(landmarks[n][1]))),
                     (int(float(landmarks[n + 1][0])), int(float(landmarks[n + 1][1]))), line_color, line_width)
        for n in range(36, 41):
            cv2.line(img, (int(float(landmarks[n][0])), int(float(landmarks[n][1]))),
                     (int(float(landmarks[n + 1][0])), int(float(landmarks[n + 1][1]))), line_color, line_width)
        cv2.line(img, (int(float(landmarks[36][0])), int(float(landmarks[36][1]))),
                 (int(float(landmarks[41][0])), int(float(landmarks[41][1]))), line_color, line_width)
        # for n in range(42, 47):
        #     cv2.line(img, (int(float(landmarks[n][0])), int(float(landmarks[n][1]))),
        #              (int(float(landmarks[n + 1][0])), int(float(landmarks[n + 1][1]))), line_color, line_width)
        # cv2.line(img, (int(float(landmarks[42][0])), int(float(landmarks[42][1]))),
        #          (int(float(landmarks[47][0])), int(float(landmarks[47][1]))), line_color, line_width)
        for n in range(48, 51):
            cv2.line(img, (int(float(landmarks[n][0])), int(float(landmarks[n][1]))),
                     (int(float(landmarks[n + 1][0])), int(float(landmarks[n + 1][1]))), line_color, line_width)

        cv2.line(img, (int(float(landmarks[51][0])), int(float(landmarks[51][1]))),
                 (int(float(landmarks[62][0])), int(float(landmarks[62][1]))), line_color, line_width)

        for n in range(60, 62):
            cv2.line(img, (int(float(landmarks[n][0])), int(float(landmarks[n][1]))),
                     (int(float(landmarks[n + 1][0])), int(float(landmarks[n + 1][1]))), line_color, line_width)

        cv2.line(img, (int(float(landmarks[60][0])), int(float(landmarks[60][1]))),
                 (int(float(landmarks[67][0])), int(float(landmarks[67][1]))), line_color, line_width)

        cv2.line(img, (int(float(landmarks[67][0])), int(float(landmarks[67][1]))),
                 (int(float(landmarks[66][0])), int(float(landmarks[66][1]))), line_color, line_width)

        cv2.line(img, (int(float(landmarks[57][0])), int(float(landmarks[57][1]))),
                 (int(float(landmarks[66][0])), int(float(landmarks[66][1]))), line_color, line_width)

        for n in range(57, 59):
            cv2.line(img, (int(float(landmarks[n][0])), int(float(landmarks[n][1]))),
                     (int(float(landmarks[n + 1][0])), int(float(landmarks[n + 1][1]))), line_color, line_width)

        cv2.line(img, (int(float(landmarks[48][0])), int(float(landmarks[48][1]))),
                 (int(float(landmarks[59][0])), int(float(landmarks[59][1]))), line_color, line_width)
    elif diff <= 30 and int(float(landmarks[30][0])) < int(float(landmarks[42][0])):

        # print('left')

        for n in range(4, 16):
            cv2.line(img, (int(float(landmarks[n][0])), int(float(landmarks[n][1]))),
                     (int(float(landmarks[n + 1][0])), int(float(landmarks[n + 1][1]))), line_color, line_width)
        # for n in range(17, 21):
        #     cv2.line(img, (int(float(landmarks[n][0])), int(float(landmarks[n][1]))),
        #              (int(float(landmarks[n + 1][0])), int(float(landmarks[n + 1][1]))), line_color, line_width)
        for n in range(22, 26):
            cv2.line(img, (int(float(landmarks[n][0])), int(float(landmarks[n][1]))),
                     (int(float(landmarks[n + 1][0])), int(float(landmarks[n + 1][1]))), line_color, line_width)
        for n in range(27, 30):
            cv2.line(img, (int(float(landmarks[n][0])), int(float(landmarks[n][1]))),
                     (int(float(landmarks[n + 1][0])), int(float(landmarks[n + 1][1]))), line_color, line_width)
        for n in range(33, 35):
            cv2.line(img, (int(float(landmarks[n][0])), int(float(landmarks[n][1]))),
                     (int(float(landmarks[n + 1][0])), int(float(landmarks[n + 1][1]))), line_color, line_width)
        # for n in range(36, 41):
        #     cv2.line(img, (int(float(landmarks[n][0])), int(float(landmarks[n][1]))),
        #              (int(float(landmarks[n + 1][0])), int(float(landmarks[n + 1][1]))), line_color, line_width)
        # cv2.line(img, (int(float(landmarks[36][0])), int(float(landmarks[36][1]))),
        #          (int(float(landmarks[41][0])), int(float(landmarks[41][1]))), line_color, line_width)
        for n in range(42, 47):
            cv2.line(img, (int(float(landmarks[n][0])), int(float(landmarks[n][1]))),
                     (int(float(landmarks[n + 1][0])), int(float(landmarks[n + 1][1]))), line_color, line_width)
        cv2.line(img, (int(float(landmarks[42][0])), int(float(landmarks[42][1]))),
                 (int(float(landmarks[47][0])), int(float(landmarks[47][1]))), line_color, line_width)

        for n in range(51, 57):
            cv2.line(img, (int(float(landmarks[n][0])), int(float(landmarks[n][1]))),
                     (int(float(landmarks[n + 1][0])), int(float(landmarks[n + 1][1]))), line_color, line_width)
        cv2.line(img, (int(float(landmarks[51][0])), int(float(landmarks[51][1]))),
                 (int(float(landmarks[62][0])), int(float(landmarks[62][1]))), line_color, line_width)
        for n in range(62, 66):
            cv2.line(img, (int(float(landmarks[n][0])), int(float(landmarks[n][1]))),
                     (int(float(landmarks[n + 1][0])), int(float(landmarks[n + 1][1]))), line_color, line_width)
        cv2.line(img, (int(float(landmarks[66][0])), int(float(landmarks[66][1]))),
                 (int(float(landmarks[57][0])), int(float(landmarks[57][1]))), line_color, line_width)

    else:


        for n in range(0, 16):
            cv2.line(img, (int(float(landmarks[n][0])), int(float(landmarks[n][1]))),
                     (int(float(landmarks[n + 1][0])), int(float(landmarks[n + 1][1]))), line_color, line_width)
        for n in range(17, 21):
            cv2.line(img, (int(float(landmarks[n][0])), int(float(landmarks[n][1]))),
                     (int(float(landmarks[n + 1][0])), int(float(landmarks[n + 1][1]))), line_color, line_width)
        for n in range(22, 26):
            cv2.line(img, (int(float(landmarks[n][0])), int(float(landmarks[n][1]))),
                     (int(float(landmarks[n + 1][0])), int(float(landmarks[n + 1][1]))), line_color, line_width)
        for n in range(27, 30):
            cv2.line(img, (int(float(landmarks[n][0])), int(float(landmarks[n][1]))),
                     (int(float(landmarks[n + 1][0])), int(float(landmarks[n + 1][1]))), line_color, line_width)
        for n in range(31, 35):
            cv2.line(img, (int(float(landmarks[n][0])), int(float(landmarks[n][1]))),
                     (int(float(landmarks[n + 1][0])), int(float(landmarks[n + 1][1]))), line_color, line_width)
        for n in range(36, 41):
            cv2.line(img, (int(float(landmarks[n][0])), int(float(landmarks[n][1]))),
                     (int(float(landmarks[n + 1][0])), int(float(landmarks[n + 1][1]))), line_color, line_width)
        cv2.line(img, (int(float(landmarks[36][0])), int(float(landmarks[36][1]))),
                 (int(float(landmarks[41][0])), int(float(landmarks[41][1]))), line_color, line_width)
        for n in range(42, 47):
            cv2.line(img, (int(float(landmarks[n][0])), int(float(landmarks[n][1]))),
                     (int(float(landmarks[n + 1][0])), int(float(landmarks[n + 1][1]))), line_color, line_width)
        cv2.line(img, (int(float(landmarks[42][0])), int(float(landmarks[42][1]))),
                 (int(float(landmarks[47][0])), int(float(landmarks[47][1]))), line_color, line_width)
        for n in range(48, 59):
            cv2.line(img, (int(float(landmarks[n][0])), int(float(landmarks[n][1]))),
                     (int(float(landmarks[n + 1][0])), int(float(landmarks[n + 1][1]))), line_color, line_width)
        cv2.line(img, (int(float(landmarks[48][0])), int(float(landmarks[48][1]))),
                 (int(float(landmarks[59][0])), int(float(landmarks[59][1]))), line_color, line_width)
        for n in range(60, 67):
            cv2.line(img, (int(float(landmarks[n][0])), int(float(landmarks[n][1]))),
                     (int(float(landmarks[n + 1][0])), int(float(landmarks[n + 1][1]))), line_color, line_width)
        cv2.line(img, (int(float(landmarks[60][0])), int(float(landmarks[60][1]))),
                 (int(float(landmarks[67][0])), int(float(landmarks[67][1]))), line_color, line_width)

    # cv2.imshow('img_crop', img)
    # cv2.imshow('img_show', img_show)
    # cv2.waitKey(0)
    # print(shape)
    # assert False
    return img









def tran_point(point, M):
    pts = np.float32(point).reshape([-1,2])
    pts = np.hstack([pts,np.ones([len(pts),1])]).T
    target_point = np.dot(M,pts)
    return target_point.squeeze()

def get_arcface(rimg, shape):
    landmark = []
    # print(img)
    # for lm in range(68):
    #     w, h = landmark[lm, 0], landmark[lm, 1]
    #     shape.append([w, h])

    for p in shape[0]:
        w, h = p[0], p[1]
        landmark.append([w, h])

    src = np.array([
      [30.2946*2, 51.6963*2],
      [65.5318*2, 51.5014*2],
      [48.0252*2, 71.7366*2],
      [33.5493*2, 92.3655*2],
      [62.7299*2, 92.2041*2] ], dtype=np.float32 )
    src[:,0] += 31.9721

    # src = np.array([
    #   [30.2946*2.2, 51.6963*2.2],
    #   [65.5318*2.2, 51.5014*2.2],
    #   [48.0252*2.2, 71.7366*2.2],
    #   [33.5493*2.2, 92.3655*2.2],
    #   [62.7299*2.2, 92.2041*2.2] ], dtype=np.float32 )
    #
    # src[:,0] += 8 * 2.2
    # src[:, 1] -= 30

    # src = np.array([
    #   [30.2946, 51.6963],
    #   [65.5318, 51.5014],
    #   [48.0252, 71.7366],
    #   [33.5493, 92.3655],
    #   [62.7299, 92.2041] ], dtype=np.float32 )
    # src[:,0] += 8.0
    # src = np.array([
    #   [69.2448, 118.1629],
    #   [149.7869, 117.7174],
    #   [109.7718, 163.9693],
    #   [76.6841, 211.1211],
    #   [154.8112, 210.7522] ], dtype=np.float32 )
    # src[:,0] += 18.2857

    landmark = np.array(landmark)
    # print(landmark.shape[0])
    assert landmark.shape[0] == 68 or landmark.shape[0] == 5
    assert landmark.shape[1] == 2
    # print(landmark.shape[0])
    if landmark.shape[0] == 68:
        landmark5 = np.zeros((5, 2), dtype=np.float32)
        landmark5[0] = (landmark[36] + landmark[39]) / 2
        landmark5[1] = (landmark[42] + landmark[45]) / 2
        landmark5[2] = landmark[30]
        landmark5[3] = landmark[48]
        landmark5[4] = landmark[54]
    else:
        print('5')
        landmark5 = landmark
    # print(landmark5)
    tform = trans.SimilarityTransform()
    tform.estimate(landmark5, src)
    # print(src[:,0])
    M = tform.params[0:2, :]
    img_crop = cv2.warpAffine(rimg, M, (256, 256), borderValue=0.0)

    for i in range(68):
        landmark[i] = tran_point(landmark[i], M)

    img_shape = drawshape(landmark)


    LM = []
    for LM_crop in landmark:

        LM.append(LM_crop[0]/256)
        LM.append(LM_crop[1]/256)

    # cv2.imshow('rimg', img_crop)
    # cv2.imshow('img', img_shape)
    # cv2.waitKey(0)
    # assert False

    return img_crop, img_shape, LM

img_crop = cv2.imread('./for_demo/0000375.jpg')
img_crop_2 = cv2.imread('./for_demo/0000425.jpg')
img_crop_3 = cv2.imread('./for_demo/0005000.jpg')
img_crop_4 = cv2.imread('./for_demo/0002525.jpg')


img_crop = cv2.resize(img_crop, (256, 256))
img_crop_2 = cv2.resize(img_crop_2, (256, 256))
img_crop_3 = cv2.resize(img_crop_3, (256, 256))
img_crop_4 = cv2.resize(img_crop_4, (256, 256))

source = toTensor(img_crop)
source_2 = toTensor(img_crop_2)
source_3 = toTensor(img_crop_3)
source_4 = toTensor(img_crop_4)




source_all = np.zeros((4, 3, 256, 256))

source_all = torch.from_numpy(source_all)

source_all= source_all.float()

source_all[0,:,:,:] = source
source_all[1,:,:,:] = source_2
source_all[2,:,:,:] = source_3
source_all[3,:,:,:] = source_4


source_style_code = solver.extract(source_all)




nn = 0
while (cap.isOpened()):
    # start = timeit.default_timer()
    ret, frame = cap.read()

    try:
        preds = fa.get_landmarks(frame)
        # print(preds)

        img_crop2, img_shape, shape = get_arcface(frame, preds)

        # print(shape)

        shape = torch.FloatTensor(shape)

        shape_all = np.zeros((4, 136))

        shape_all = torch.from_numpy(shape_all)

        shape_all = shape_all.float()

        shape_all[0, :] = shape
        shape_all[1, :] = shape
        shape_all[2, :] = shape
        shape_all[3, :] = shape



        lm_fake, lm_fake_2, lm_fake_3, lm_fake_4 = solver_2.sample(source_all, shape_all)

        lm_shape_fake = toTensor(lm_fake)
        lm_shape_fake_2 = toTensor(lm_fake_2)
        lm_shape_fake_3 = toTensor(lm_fake_3)
        lm_shape_fake_4 = toTensor(lm_fake_4)

        reference_all = np.zeros((4, 3, 256, 256))

        reference_all = torch.from_numpy(reference_all)

        reference_all = reference_all.float()

        reference_all[0, :, :, :] = lm_shape_fake
        reference_all[1, :, :, :] = lm_shape_fake_2
        reference_all[2, :, :, :] = lm_shape_fake_3
        reference_all[3, :, :, :] = lm_shape_fake_4


        img_fake, img_fake_2, img_fake_3, img_fake_4 = solver.sample(source_style_code, reference_all)
        #
        img_fake = cv2.cvtColor(img_fake, cv2.COLOR_RGB2BGR)
        img_fake_2 = cv2.cvtColor(img_fake_2, cv2.COLOR_RGB2BGR)
        img_fake_3 = cv2.cvtColor(img_fake_3, cv2.COLOR_RGB2BGR)
        img_fake_4 = cv2.cvtColor(img_fake_4, cv2.COLOR_RGB2BGR)


        # all = np.zeros((768, 1024, 3), np.uint8)
        #
        # all[:256, 384:640, :] = img_crop2
        #
        #
        # all[256:512, :256, :] = img_crop
        # all[256:512, 256:512, :] = img_crop_2
        # all[256:512, 512:768, :] = img_crop_3
        # all[256:512, 768:1024, :] = img_crop_4
        #
        #
        # all[512:, :256, :] = img_fake
        # all[512:, 256:512, :] = img_fake_2
        # all[512:, 512:768, :] = img_fake_3
        # all[512:, 768:1024, :] = img_fake_4


        all = np.zeros((1024, 1024, 3), np.uint8)

        all[:256, 384:640, :] = img_crop2


        all[256:512, :256, :] = img_crop
        all[256:512, 256:512, :] = img_crop_2
        all[256:512, 512:768, :] = img_crop_3
        all[256:512, 768:1024, :] = img_crop_4


        all[512:768, :256, :] = lm_fake*255
        all[512:768, 256:512, :] = lm_fake_2*255
        all[512:768, 512:768, :] = lm_fake_3 * 255
        all[512:768, 768:1024, :] = lm_fake_4 * 255


        all[768:, :256, :] = img_fake
        all[768:, 256:512, :] = img_fake_2
        all[768:, 512:768, :] = img_fake_3
        all[768:, 768:1024, :] = img_fake_4





        # cv2.imshow('img_shape', lm_fake*255)
        # cv2.imshow('img_shape_2', lm_fake_2*255)
        # cv2.imshow('img_shape_3', lm_fake_3 * 255)
        # cv2.imshow('img_shape_4', lm_fake_4 * 255)

        out.write(all)
        cv2.imshow('demo', all)

    except:
        # all = np.zeros((768, 1024, 3), np.uint8)

        all = np.zeros((1024, 1024, 3), np.uint8)

        # all[:256, 384:640, :] = img_crop2

        out.write(all)
        cv2.imshow('demo', all)

        # cv2.imshow('img_ref', img_crop2)

    flag = cv2.waitKey(1)


# press "c" to save the result
    if flag == 67 or flag == 99:

        cv2.imwrite('./result/img_fake{}.png'.format(nn), img_fake)
        cv2.imwrite('./result/img_crop{}.png'.format(nn), img_crop2)
        cv2.imwrite('./result/img_fake_ori{}.png'.format(nn), img_fake_2)

        # save_image(lm_shape_fake[0], 1, './result/img_shape_fake{}.png'.format(nn))
        cv2.imwrite('./result/img_shape_fake{}.png'.format(nn), lm_fake*255)
        cv2.imwrite('./result/img_shape_ori{}.png'.format(nn), img_shape)

        nn += 1


    if flag == 27:
        print("Quit !!")



        cap.release()
        cv2.destroyAllWindows()
        break

cap.release()
out.release()
cv2.destroyAllWindows()

