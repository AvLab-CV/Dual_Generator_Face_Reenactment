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
import time
import datetime
from munch import Munch
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.model_lm_talking import build_model
from core.checkpoint import CheckpointIO
from core.data_loader_lm_perceptual import InputFetcher
import core.utils_lm as utils
from metrics.eval import calculate_metrics
from tensorboardX import SummaryWriter

from ms1m_ir50.model_irse import IR_50
from scipy import spatial

import network
from network.vgg import vgg_face, VGG_Activations
from torchvision.models import vgg19
import FR_Pretrained_Test
from FR_Pretrained_Test.Model.model_lightcnn import LightCNN_9Layers, LightCNN_29Layers, LightCNN_29Layers_v2




class Solver(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.nets, self.nets_ema = build_model(args)

        self.writer = SummaryWriter('log/train_list_celebv_finetune_id_cyc')

        for name, module in self.nets.items():
            utils.print_network(module, name)
            setattr(self, name, module)
        for name, module in self.nets_ema.items():
            setattr(self, name + '_ema', module)

        if args.mode == 'train':
            self.optims = Munch()
            for net in self.nets.keys():
                if net == 'fan':
                    continue
                self.optims[net] = torch.optim.Adam(
                    params=self.nets[net].parameters(),
                    lr=args.f_lr if net == 'mapping_network' else args.lr,
                    betas=[args.beta1, args.beta2],
                    weight_decay=args.weight_decay)

            self.ckptios = [
                CheckpointIO(ospj(args.checkpoint_dir, '{}_nets.ckpt'), **self.nets),
                CheckpointIO(ospj(args.checkpoint_dir, '{}_nets_ema.ckpt'), **self.nets_ema),
                CheckpointIO(ospj(args.checkpoint_dir, '{}_optims.ckpt'), **self.optims)]
        else:

            self.ckptios = [CheckpointIO(ospj(args.checkpoint_dir, '{}_nets_ema.ckpt'.format(args.resume_iter)), **self.nets_ema)]

        self.to(self.device)
        for name, network in self.named_children():
            # Do not initialize the FAN parameters
            if ('ema' not in name) and ('fan' not in name):
                print('Initializing %s...' % name)
                network.apply(utils.he_init)

    def _save_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.save(step)

    def _load_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.load(step)

    def _reset_grad(self):
        for optim in self.optims.values():
            optim.zero_grad()

    def train(self, loaders):
        if self.args.loss == 'arcface':
            BACKBONE_RESUME_ROOT = 'D:/face-recognition/stargan-v2-master/ms1m_ir50/backbone_ir50_ms1m_epoch120.pth'

            INPUT_SIZE = [112, 112]
            arcface = IR_50(INPUT_SIZE)

            if os.path.isfile(BACKBONE_RESUME_ROOT):
                arcface.load_state_dict(torch.load(BACKBONE_RESUME_ROOT))
                print("Loading Backbone Checkpoint '{}'".format(BACKBONE_RESUME_ROOT))

            DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            criterion_id = arcface.to(DEVICE)
        elif self.args.loss == 'perceptual':
            criterion_id = network.LossEG(False, 0)
        elif self.args.loss == 'lightcnn':
            BACKBONE_RESUME_ROOT = 'D:/face-reenactment/stargan-v2-master/FR_Pretrained_Test/Pretrained/LightCNN/LightCNN_29Layers_V2_checkpoint.pth.tar'


            Model = LightCNN_29Layers_v2()

            if os.path.isfile(BACKBONE_RESUME_ROOT):

                Model = WrappedModel(Model)
                checkpoint = torch.load(BACKBONE_RESUME_ROOT)
                Model.load_state_dict(checkpoint['state_dict'])
                print("Loading Backbone Checkpoint '{}'".format(BACKBONE_RESUME_ROOT))

            DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            criterion_id = Model.to(DEVICE)

        if self.args.id_embed:
            id_embedder = network.vgg_feature(False, 0)
        else:
            id_embedder = None

        if self.args.fea_match:
            fea_match = network.vgg_feature(False, 0)
        else:
            fea_match = None



        args = self.args
        nets = self.nets
        nets_ema = self.nets_ema
        optims = self.optims

        # fetch random validation images for debugging
        fetcher = InputFetcher(loaders.src, args.latent_dim, 'train',args.multi)
        fetcher_val = InputFetcher(loaders.val, args.latent_dim, 'val', args.multi)
        inputs_val = next(fetcher_val)

        # resume training if necessary
        if args.resume_iter > 0:
            self._load_checkpoint(args.resume_iter)

        # remember the initial value of ds weight
        initial_lambda_ds = args.lambda_ds

        print('Start training...')
        start_time = time.time()
        for i in range(args.resume_iter, args.total_iters):
            # fetch images and labels
            inputs = next(fetcher)

            if args.multi:
                x1_1_source = inputs.x1
                x1_2_source = inputs.x2
                x1_3_source = inputs.x3
                x1_4_source = inputs.x4
                x1_5_source = inputs.x5
                x1_6_source = inputs.x6
                x1_7_source = inputs.x7
                x1_8_source = inputs.x8



                x9_target, x9_target_lm = inputs.x9, inputs.x9_lm


                d_loss, d_losses = compute_d_loss_multi(
                    nets, args, x1_1_source, x1_2_source, x1_3_source, x1_4_source, x1_5_source, x1_6_source, x1_7_source, x1_8_source, x9_target, x9_target_lm, masks=None, loss_select=args.loss)
                self._reset_grad()
                d_loss.backward()
                optims.discriminator.step()

                # train the generator
                g_loss, g_losses = compute_g_loss_multi(
                    nets, args,x1_1_source, x1_2_source, x1_3_source, x1_4_source, x1_5_source, x1_6_source, x1_7_source, x1_8_source, x9_target, x9_target_lm, criterion_id, masks=None,
                    loss_select=args.loss)
                self._reset_grad()
                g_loss.backward()
                if args.id_embed:
                    optims.generator.step()
                    optims.mlp.step()
                else:
                    optims.generator.step()
                    optims.style_encoder.step()


            else:
                x1_source, x1_source_lm = inputs.x1, inputs.x_lm
                x2_target, x2_target_lm = inputs.x2, inputs.x2_lm

                if args.masks:

                    masks = x2_target_lm
                else:
                    masks = None

                # train the discriminator
                if args.pix2pix:
                    d_loss, d_losses = compute_d_loss(
                        nets, args, x1_source,x1_source_lm, x2_target, x2_target_lm, masks=masks, loss_select = args.loss, embedder = id_embedder)
                    self._reset_grad()
                    d_loss.backward()
                    optims.discriminator.step()
                    optims.discriminator2.step()

                else:
                    d_loss, d_losses = compute_d_loss(
                        nets, args, x1_source,x1_source_lm, x2_target, x2_target_lm, masks=masks, loss_select = args.loss, embedder = id_embedder)
                    self._reset_grad()
                    d_loss.backward()
                    optims.discriminator.step()



                # train the generator
                g_loss, g_losses = compute_g_loss(
                    nets, args, x1_source, x1_source_lm, x2_target, x2_target_lm, criterion_id, masks=masks, loss_select = args.loss, embedder = id_embedder, fea=fea_match)
                self._reset_grad()
                g_loss.backward()
                if args.id_embed:
                    optims.generator.step()
                    optims.mlp.step()
                else:
                    optims.generator.step()
                    optims.style_encoder.step()




            # compute moving average of network parameters
            moving_average(nets.generator, nets_ema.generator, beta=0.999)
            # moving_average(nets.mapping_network, nets_ema.mapping_network, beta=0.999)
            if args.id_embed:
                moving_average(nets.mlp, nets_ema.mlp, beta=0.999)
            else:
                moving_average(nets.style_encoder, nets_ema.style_encoder, beta=0.999)

            # decay weight for diversity sensitive loss
            if args.lambda_ds > 0:
                args.lambda_ds -= (initial_lambda_ds / args.ds_iter)

            # print out log info
            if (i+1) % args.print_every == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
                log = "Elapsed time [%s], Iteration [%i/%i], " % (elapsed, i+1, args.total_iters)
                all_losses = dict()
                for loss, prefix in zip([d_losses, g_losses],
                                        ['D/', 'G/']):
                    for key, value in loss.items():
                        all_losses[prefix + key] = value
                all_losses['G/lambda_ds'] = args.lambda_ds
                log += ' '.join(['%s: [%.4f]' % (key, value) for key, value in all_losses.items()])
                print(log)
                for key, value in all_losses.items():
                    self.writer.add_scalar(key, value, i+1)

            # generate images for debugging
            if (i+1) % args.sample_every == 0:
                os.makedirs(args.sample_dir, exist_ok=True)

                utils.debug_image(nets_ema, args, inputs=inputs_val, step=i+1, embedder = id_embedder)

            # save model checkpoints
            if (i+1) % args.save_every == 0:
                self._save_checkpoint(step=i+1)

            # compute FID and LPIPS if necessary
            if (i+1) % args.eval_every == 0:
                calculate_metrics(nets_ema, args, i+1, mode='latent')
                # calculate_metrics(nets_ema, args, i+1, mode='reference')
        self.writer.close()

    @torch.no_grad()
    def sample(self, loaders):

        if self.args.id_embed:
            id_embedder = network.vgg_feature(False, 0)
        else:
            id_embedder = None

        args = self.args
        nets_ema = self.nets_ema
        os.makedirs(args.result_dir, exist_ok=True)
        self._load_checkpoint(args.resume_iter)

        src = next(InputFetcher(loaders.src, args.latent_dim, 'test'))
        # ref = next(InputFetcher(loaders.ref, None, args.latent_dim, 'test'))

        fname = ospj(args.result_dir, 'reconstruct.jpg')
        print('Working on {}...'.format(fname))
        # utils.translate_and_reconstruct_sample(nets_ema, args, src.x1, src.x1_c, src.x2, src.x2_lm, fname, conf, arcface)
        utils.translate_and_reconstruct(nets_ema, args, src.x1, src.x2, src.x2_lm, fname, id_embedder)

        

    @torch.no_grad()
    def evaluate(self):
        args = self.args
        nets_ema = self.nets_ema
        resume_iter = args.resume_iter
        self._load_checkpoint(args.resume_iter)
        calculate_metrics(nets_ema, args, step=resume_iter, mode='latent')
        # calculate_metrics(nets_ema, args, step=resume_iter, mode='reference')


def compute_d_loss(nets, args, x1_source, x1_source_lm, x2_target, x2_target_lm, masks=None, loss_select = 'perceptual', embedder = None):
    if args.pix2pix:


        x2_target.requires_grad_()


        _, real_out_1 = nets.discriminator(x2_target, x2_target_lm)
        _, real_out_2 = nets.discriminator2(x2_target, x2_target_lm)

        real_out = real_out_1 + real_out_2
        loss_real = adv_loss(real_out, 1)


        # with fake images
        with torch.no_grad():
            s_trg = nets.style_encoder(x1_source)
            x_fake = nets.generator(x2_target_lm, s_trg, masks=masks, loss_select=loss_select)
            # x_fake_down = F.interpolate(x_fake, size=args.img_size // 2, mode='bilinear')

        _, fake_out_1 = nets.discriminator(x_fake, x2_target_lm)
        _, fake_out_2 = nets.discriminator2(x_fake, x2_target_lm)

        fake_out = fake_out_1 + fake_out_2
        loss_fake = adv_loss(fake_out, 0)

        loss = loss_real + loss_fake

        return loss, Munch(real=loss_real.item(),
                           fake=loss_fake.item())
    else:
        # with real images
        x2_target.requires_grad_()
        out = nets.discriminator(x2_target, x2_target_lm)
        loss_real = adv_loss(out, 1)
        loss_reg = r1_reg(out, x2_target)

        # with fake images
        with torch.no_grad():
            if args.id_embed:
                s_fea = embedder(x1_source)
                s_trg = nets.mlp(s_fea)
            else:
                s_trg = nets.style_encoder(x1_source)
            x_fake = nets.generator(x2_target_lm, s_trg, masks=masks, loss_select=loss_select )
        out = nets.discriminator(x_fake, x2_target_lm)
        loss_fake = adv_loss(out, 0)

        loss = loss_real + loss_fake + args.lambda_reg * loss_reg

        return loss, Munch(real=loss_real.item(),
                           fake=loss_fake.item(),
                           reg=loss_reg.item())

# def compute_g_loss(nets, args, x1_source, x2_target, x2_target_lm, arcface, conf, z_trgs=None, x_refs=None, masks=None):
def compute_g_loss(nets, args, x1_source, x1_source_lm, x2_target, x2_target_lm,criterion_id, masks=None, loss_select = 'perceptual', embedder = None, fea=None):

    # adversarial loss

    if args.id_embed:
        with torch.no_grad():
            s_fea = embedder(x1_source)
        s_trg = nets.mlp(s_fea)
    else:
        s_trg = nets.style_encoder(x1_source)
    # s_trg = nets.style_encoder(x1_source)

    if args.pix2pix:

        real_fea_1, real_out_1 = nets.discriminator(x2_target, x2_target_lm)
        real_fea_2, real_out_2 = nets.discriminator2(x2_target, x2_target_lm)

        # x2_target_lm_down = F.interpolate(x2_target_lm, size=args.img_size // 2, mode='bilinear')

        x_fake = nets.generator(x2_target_lm, s_trg, masks=masks, loss_select=loss_select)
        # x_fake_down = F.interpolate(x_fake, size=args.img_size // 2, mode='bilinear')
        fake_fea_1, fake_out_1 = nets.discriminator(x_fake, x2_target_lm)
        fake_fea_2, fake_out_2 = nets.discriminator2(x_fake, x2_target_lm)

        # out_1 = nets.discriminator(x_fake, x2_target_lm)
        # out_2 = nets.discriminator2(x_fake, x2_target_lm)

        out = fake_out_1 + fake_out_2
        loss_adv = adv_loss(out, 1)

        for num in range(6):
            if num == 0:
                loss_fm_1 =  torch.mean(torch.abs(fake_fea_1[num] - real_fea_1[num]))
                loss_fm_2 =  torch.mean(torch.abs(fake_fea_2[num] - real_fea_2[num]))
            else:
                loss_fm_1 +=  torch.mean(torch.abs(fake_fea_1[num] - real_fea_1[num]))
                loss_fm_2 +=  torch.mean(torch.abs(fake_fea_2[num] - real_fea_2[num]))
        loss_fm = loss_fm_1 + loss_fm_2


    else:
        x_fake = nets.generator(x2_target_lm, s_trg, masks=masks, loss_select=loss_select)
        out = nets.discriminator(x_fake, x2_target_lm)
        loss_adv = adv_loss(out, 1)


    if args.l2:
        loss_pixel_1 = torch.mean(F.mse_loss(x_fake, x2_target))
    else:
        loss_pixel_1 = torch.mean(torch.abs(x_fake - x2_target))
    # loss_cyc_1 = torch.mean(F.mse_loss(x_fake, x2_target))
    # loss_cyc_2 = torch.mean(torch.abs(x_fake_2 - x1_source))

    # loss_cyc = loss_cyc_1 + loss_cyc_2

    # loss_cyc = torch.mean(torch.abs(x_fake - x2_target))
    if args.loss == 'arcface':
        x1_source = nn.functional.interpolate(x1_source[:, :, :, :], size=(112, 112), mode='bilinear')
        x_fake = nn.functional.interpolate(x_fake[:, :, :, :], size=(112, 112), mode='bilinear')
        x2_target = nn.functional.interpolate(x2_target[:, :, :, :], size=(112, 112), mode='bilinear')

        criterion_id.eval()
        with torch.torch.no_grad():
            source_embs = criterion_id(x1_source)
            target_embs = criterion_id(x2_target)
            fake_embs = criterion_id(x_fake)
        print(source_embs)
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        output = cos(target_embs, fake_embs)
        loss_id = torch.mean(1 - output)
        output2 = cos(target_embs, source_embs)
        loss_id2 = torch.mean(1 - output2)
        print(output)
        print(output2)
        print('loss_id: {}'.format(loss_id))
        print('loss_id2: {}'.format(loss_id2))
        assert False


    elif args.loss == 'perceptual':
        loss_id = criterion_id(x_fake, x2_target)
    elif args.loss == 'lightcnn':
        # x1_source = nn.functional.interpolate(x1_source[:, :, :, :], size=(128, 128), mode='bilinear')
        x_fake = nn.functional.interpolate(x_fake[:, :, :, :], size=(128, 128), mode='bilinear')
        x2_target = nn.functional.interpolate(x2_target[:, :, :, :], size=(128, 128), mode='bilinear')

        def rgb2gray(img):
            return img[:, 0, :, :] * 0.2989 + img[:, 1, :, :] * 0.5870 + img[:, 2, :, :] * 0.1140

        x_fake = rgb2gray(x_fake)
        x2_target = rgb2gray(x2_target)


        # print(x1_source.size())

        criterion_id.eval()
        with torch.torch.no_grad():
            _ = criterion_id(x1_source)
            try:
                source_embs = criterion_id.feature
            except:
                source_embs = criterion_id.module.feature
            _ = criterion_id(x2_target)
            try:
                target_embs = criterion_id.feature
            except:
                target_embs = criterion_id.module.feature
            _ = criterion_id(x_fake)
            try:
                fake_embs = criterion_id.feature
            except:
                fake_embs = criterion_id.module.feature



        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        # arr3 =
        # print(source_embs[0])
        # print(source_embs[0].size())

        output = cos(target_embs, fake_embs)
        # print(output)
        loss_id = torch.mean(1 - output)
        # output2 = cos(target_embs, source_embs)
        # loss_id2 = torch.mean(1 - output2)
        # print(output)
        # print(output2)
        #
        # print('loss_id: {}'.format(loss_id))
        # print('loss_id2: {}'.format(loss_id2))
        # assert False



    # # loss = loss_adv  + args.lambda_cyc * loss_cyc_1 + args.lambda_cyc * loss_cyc_2 + args.lambda_con * loss_con + args.lambda_id * loss_id_1 + args.lambda_id * loss_id_2
    #
    # loss = loss_adv + args.lambda_pixel * loss_pixel_1 + args.lambda_id * loss_id
    # # return loss, Munch(adv=loss_adv.item(),
    # #                    cyc_1=loss_cyc_1.item(),
    # #                    cyc_2=loss_cyc_2.item(),
    # #                    con=loss_con.item(),
    # #                    id_1=loss_id_1.item(),
    # #                    id_2=loss_id_2.item())
    # return loss, Munch(adv=loss_adv.item(),
    #                    pixel_1=loss_pixel_1.item(),
    #                    id=loss_id.item())


    if args.fea_match:

        with torch.no_grad():
            target_fea = fea(x2_target)
            fake_fea = fea(x_fake)

        loss_fea_match = torch.mean(torch.abs(fake_fea - target_fea))
        loss = loss_adv + args.lambda_pixel * loss_pixel_1 + args.lambda_id * loss_id + args.lambda_fea_match * loss_fea_match

        return loss, Munch(adv=loss_adv.item(),
                        pixel_1=loss_pixel_1.item(),
                        id=loss_id.item(), fea_match=loss_fea_match.item())

    elif args.id_cyc:
        if args.pix2pix:
            s_trg_2 = nets.style_encoder(x_fake)


            loss_id_cyc = torch.mean(torch.abs(s_trg_2 - s_trg))
            loss = loss_adv + args.lambda_pixel * loss_pixel_1 + args.lambda_id * loss_id+ args.lambda_id_cyc * loss_id_cyc + args.lambda_fm *loss_fm
            return loss, Munch(adv=loss_adv.item(),
                            pixel_1=loss_pixel_1.item(),
                            id=loss_id.item(), id_cyc=loss_id_cyc.item(),fm=loss_fm.item())
        else:
            s_trg_2 = nets.style_encoder(x_fake)
            # s_trg = nets.style_encoder(x1_source)

            loss_id_cyc = torch.mean(torch.abs(s_trg_2 - s_trg))
            loss = loss_adv + args.lambda_pixel * loss_pixel_1 + args.lambda_id * loss_id + args.lambda_id_cyc * loss_id_cyc

            return loss, Munch(adv=loss_adv.item(),
                            pixel_1=loss_pixel_1.item(),
                            id=loss_id.item(), id_cyc=loss_id_cyc.item())

    elif args.pix2pix:

        loss = loss_adv + args.lambda_pixel * loss_pixel_1 + args.lambda_id * loss_id + args.lambda_fm *loss_fm
        return loss, Munch(adv=loss_adv.item(),
                        pixel_1=loss_pixel_1.item(),
                        id=loss_id.item(),fm=loss_fm.item())
    else:

    # loss = loss_adv  + args.lambda_cyc * loss_cyc_1 + args.lambda_cyc * loss_cyc_2 + args.lambda_con * loss_con + args.lambda_id * loss_id_1 + args.lambda_id * loss_id_2

        loss = loss_adv + args.lambda_pixel * loss_pixel_1 + args.lambda_id * loss_id
    # return loss, Munch(adv=loss_adv.item(),
    #                    cyc_1=loss_cyc_1.item(),
    #                    cyc_2=loss_cyc_2.item(),
    #                    con=loss_con.item(),
    #                    id_1=loss_id_1.item(),
    #                    id_2=loss_id_2.item())
        return loss, Munch(adv=loss_adv.item(),
                        pixel_1=loss_pixel_1.item(),
                        id=loss_id.item())


def compute_d_loss_multi(nets, args, x1_1_source, x1_2_source, x1_3_source, x1_4_source, x1_5_source, x1_6_source, x1_7_source, x1_8_source, x9_target, x9_target_lm, masks=None, loss_select = 'perceptual'):

    # with real images
    x9_target.requires_grad_()
    out = nets.discriminator(x9_target, x9_target_lm)
    loss_real = adv_loss(out, 1)
    loss_reg = r1_reg(out, x9_target)

    # with fake images
    with torch.no_grad():
        s_trg = nets.style_encoder(x1_1_source)
        # print(s_trg)
        s_trg += nets.style_encoder(x1_2_source)
        s_trg += nets.style_encoder(x1_3_source)
        s_trg += nets.style_encoder(x1_4_source)
        s_trg += nets.style_encoder(x1_5_source)
        s_trg += nets.style_encoder(x1_6_source)
        s_trg += nets.style_encoder(x1_7_source)
        s_trg += nets.style_encoder(x1_8_source)

        s_trg_mean = s_trg/8



        x_fake = nets.generator(x9_target_lm, s_trg_mean, masks=masks, loss_select=loss_select)


    out = nets.discriminator(x_fake, x9_target_lm)
    loss_fake = adv_loss(out, 0)

    loss = loss_real + loss_fake + args.lambda_reg * loss_reg

    return loss, Munch(real=loss_real.item(),
                       fake=loss_fake.item(),
                       reg=loss_reg.item())

# def compute_g_loss(nets, args, x1_source, x2_target, x2_target_lm, arcface, conf, z_trgs=None, x_refs=None, masks=None):
def compute_g_loss_multi(nets, args, x1_1_source, x1_2_source, x1_3_source, x1_4_source, x1_5_source, x1_6_source, x1_7_source, x1_8_source, x9_target, x9_target_lm,criterion_id, masks=None, loss_select = 'perceptual'):

    # adversarial loss
    s_trg = nets.style_encoder(x1_1_source)
    s_trg += nets.style_encoder(x1_2_source)
    s_trg += nets.style_encoder(x1_3_source)
    s_trg += nets.style_encoder(x1_4_source)
    s_trg += nets.style_encoder(x1_5_source)
    s_trg += nets.style_encoder(x1_6_source)
    s_trg += nets.style_encoder(x1_7_source)
    s_trg += nets.style_encoder(x1_8_source)

    s_trg_mean = s_trg/8

    x_fake = nets.generator(x9_target_lm, s_trg_mean, masks=masks, loss_select=loss_select)
    out = nets.discriminator(x_fake, x9_target_lm)
    loss_adv = adv_loss(out, 1)

    loss_pixel_1 = torch.mean(torch.abs(x_fake - x9_target))

    # loss_cyc = torch.mean(torch.abs(x_fake - x2_target))
    if args.loss == 'arcface':
        x1_source = nn.functional.interpolate(x1_source[:, :, :, :], size=(112, 112), mode='bilinear')
        x_fake = nn.functional.interpolate(x_fake[:, :, :, :], size=(112, 112), mode='bilinear')
        x2_target = nn.functional.interpolate(x2_target[:, :, :, :], size=(112, 112), mode='bilinear')

        criterion_id.eval()
        with torch.torch.no_grad():
            source_embs = criterion_id(x1_source)
            target_embs = criterion_id(x2_target)
            fake_embs = criterion_id(x_fake)
        print(source_embs)
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        output = cos(target_embs, fake_embs)
        loss_id = torch.mean(1 - output)
        output2 = cos(target_embs, source_embs)
        loss_id2 = torch.mean(1 - output2)
        print(output)
        print(output2)
        print('loss_id: {}'.format(loss_id))
        print('loss_id2: {}'.format(loss_id2))
        assert False


    elif args.loss == 'perceptual':
        loss_id = criterion_id(x_fake, x9_target)
    elif args.loss == 'lightcnn':
        x1_source = nn.functional.interpolate(x1_source[:, 0, :, :], size=(128, 128), mode='bilinear')
        x_fake = nn.functional.interpolate(x_fake[:, 0, :, :], size=(128, 128), mode='bilinear')
        x2_target = nn.functional.interpolate(x2_target[:, 0, :, :], size=(128, 128), mode='bilinear')

        criterion_id.eval()
        with torch.torch.no_grad():
            source_embs = criterion_id(x1_source)
            target_embs = criterion_id(x2_target)
            fake_embs = criterion_id(x_fake)
        print(source_embs)
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        output = cos(target_embs, fake_embs)
        loss_id = torch.mean(1 - output)
        output2 = cos(target_embs, source_embs)
        loss_id2 = torch.mean(1 - output2)
        print(output)
        print(output2)
        print('loss_id: {}'.format(loss_id))
        print('loss_id2: {}'.format(loss_id2))
        assert False



    # loss = loss_adv  + args.lambda_cyc * loss_cyc_1 + args.lambda_cyc * loss_cyc_2 + args.lambda_con * loss_con + args.lambda_id * loss_id_1 + args.lambda_id * loss_id_2

    loss = loss_adv + args.lambda_pixel * loss_pixel_1 + args.lambda_id * loss_id
    # return loss, Munch(adv=loss_adv.item(),
    #                    cyc_1=loss_cyc_1.item(),
    #                    cyc_2=loss_cyc_2.item(),
    #                    con=loss_con.item(),
    #                    id_1=loss_id_1.item(),
    #                    id_2=loss_id_2.item())
    return loss, Munch(adv=loss_adv.item(),
                       pixel_1=loss_pixel_1.item(),
                       id=loss_id.item())





def moving_average(model, model_test, beta=0.999):
    for param, param_test in zip(model.parameters(), model_test.parameters()):
        param_test.data = torch.lerp(param.data, param_test.data, beta)


def adv_loss(logits, target):
    assert target in [1, 0]
    targets = torch.full_like(logits, fill_value=target)
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss


def r1_reg(d_out, x_in):
    # zero-centered gradient penalty for real images
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg



def load_arcface_2():

    BACKBONE_RESUME_ROOT = 'D:/face-recognition/stargan-v2-master/ms1m_ir50/backbone_ir50_ms1m_epoch120.pth'

    INPUT_SIZE = [112, 112]
    BACKBONE = IR_50(INPUT_SIZE)

    if os.path.isfile(BACKBONE_RESUME_ROOT):

        BACKBONE.load_state_dict(torch.load(BACKBONE_RESUME_ROOT))
        print("Loading Backbone Checkpoint '{}'".format(BACKBONE_RESUME_ROOT))

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    BACKBONE = BACKBONE.to(DEVICE)
    BACKBONE.eval()
    # for param in BACKBONE.parameters():
    #     param.requires_grad = False
    # print(BACKBONE)

    return BACKBONE

def extract_fea(data):

    BACKBONE_RESUME_ROOT = 'D:/face-recognition/stargan-v2-master/ms1m_ir50/backbone_ir50_ms1m_epoch120.pth'

    INPUT_SIZE = [112, 112]
    BACKBONE = IR_50(INPUT_SIZE)

    if os.path.isfile(BACKBONE_RESUME_ROOT):

        BACKBONE.load_state_dict(torch.load(BACKBONE_RESUME_ROOT))
        print("Loading Backbone Checkpoint '{}'".format(BACKBONE_RESUME_ROOT))

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    BACKBONE = BACKBONE.to(DEVICE)
    BACKBONE.eval()
    with torch.no_grad():
        fea = BACKBONE(data)

    return fea

class WrappedModel(nn.Module):
    def __init__(self, module):
        super(WrappedModel, self).__init__()
        self.module = module
    def forward(self, x):
        return self.module(x)
