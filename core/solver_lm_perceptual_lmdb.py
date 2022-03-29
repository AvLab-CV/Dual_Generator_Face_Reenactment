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
from core.data_loader_lm_perceptual_lmdb import InputFetcher
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
        # self.arcface, self.conf = load_arcface()
        # self.arcface = load_arcface_2()
        self.writer = SummaryWriter('log/test_vox_256_smalldata_id_1_20_20_retrain_alldata_id_embedder_vggface_add_noise')
        # print(self.arcface)
        # assert False
        # below setattrs are to make networks be children of Solver, e.g., for self.to(self.device)
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

            # self.ckptios = [
            #     CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets.ckpt'), **self.nets),
            #     CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets_ema.ckpt'), **self.nets_ema),
            #     CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_optims.ckpt'), **self.optims)]
            self.ckptios = [
                CheckpointIO(ospj(args.checkpoint_dir, '{}_nets.ckpt'), **self.nets),
                CheckpointIO(ospj(args.checkpoint_dir, '{}_nets_ema.ckpt'), **self.nets_ema),
                CheckpointIO(ospj(args.checkpoint_dir, '{}_optims.ckpt'), **self.optims)]
        else:

            self.ckptios = [CheckpointIO(ospj(args.checkpoint_dir, '{}_nets_ema.ckpt'.format(args.resume_iter)), **self.nets_ema)]
            # self.ckptios = [CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets_ema.ckpt'), **self.nets_ema)]

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

            # INPUT_SIZE = [128, 128]
            Model = LightCNN_29Layers_v2()

            if os.path.isfile(BACKBONE_RESUME_ROOT):

                Model = WrappedModel(Model)
                checkpoint = torch.load(BACKBONE_RESUME_ROOT)
                Model.load_state_dict(checkpoint['state_dict'])
                print("Loading Backbone Checkpoint '{}'".format(BACKBONE_RESUME_ROOT))

            DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            criterion_id = Model.to(DEVICE)
            # criterion_id = FR_Pretrained_Test.LossEG(False, 0)
        if self.args.id_embed:
            id_embedder = network.vgg_feature(False, 0)
        else:
            id_embedder = None





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



                x2_target, x2_target_lm = inputs.x5, inputs.x5_lm


                d_loss, d_losses = compute_d_loss_multi(
                    nets, args, x1_1_source, x1_2_source, x1_3_source, x1_4_source, x2_target, x2_target_lm, masks=None, loss_select=args.loss)
                self._reset_grad()
                d_loss.backward()
                optims.discriminator.step()

                # train the generator
                g_loss, g_losses = compute_g_loss_multi(
                    nets, args,x1_1_source, x1_2_source, x1_3_source, x1_4_source, x2_target, x2_target_lm, criterion_id, masks=None,
                    loss_select=args.loss)
                self._reset_grad()
                g_loss.backward()
                if args.id_embed:
                    optims.generator.step()
                    optims.mlp.step()
                else:
                    optims.generator.step()


            else:
                x1_source_lm = None
                x1_source = inputs.x1
                x2_target, x2_target_lm = inputs.x2, inputs.x2_lm
                # label = inputs.label


                # masks = nets.fan.get_heatmap(x_real) if args.w_hpf > 0 else None

                # train the discriminator
                d_loss, d_losses = compute_d_loss(
                    nets, args, x1_source,x1_source_lm, x2_target, x2_target_lm, masks=None, loss_select = args.loss, embedder = id_embedder)
                self._reset_grad()
                d_loss.backward()
                optims.discriminator.step()



                # train the generator
                g_loss, g_losses = compute_g_loss(
                    nets, args, x1_source, x1_source_lm, x2_target, x2_target_lm, criterion_id, masks=None, loss_select = args.loss, embedder = id_embedder)
                self._reset_grad()
                g_loss.backward()
                if args.id_embed:
                    optims.generator.step()
                    optims.mlp.step()
                else:
                    optims.generator.step()




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
        utils.translate_and_reconstruct(nets_ema, args, src.x1, src.x_lm, src.x2, src.x2_lm, fname, id_embedder)
        # utils.translate_and_reconstruct_sample(nets_ema, args, src.x1, src.x1_c, src.x2, src.x2_lm, fname, conf,
        #                                        arcface)

        # fname = ospj(args.result_dir, 'video_rec.mp4')
        # print('Working on {}...'.format(fname))
        # utils.video_rec(nets_ema, args, src.x1, src.x2, src.x2_lm, fname)




        # fname = ospj(args.result_dir, 'reference.jpg')
        # print('Working on {}...'.format(fname))
        # utils.translate_using_reference(nets_ema, args, src.x, ref.x, ref.y, fname)
        #
        # fname = ospj(args.result_dir, 'video_ref.mp4')
        # print('Working on {}...'.format(fname))
        # utils.video_ref(nets_ema, args, src.x, ref.x, ref.y, fname)
        #
        # N = src.x.size(0)
        #
        # y_trg_list = [torch.tensor(y).repeat(N).to(device)
        #               for y in range(min(args.num_domains, 5))]
        # z_trg_list = torch.randn(args.num_outs_per_domain, 1, args.latent_dim).repeat(1, N, 1).to(device)
        # for psi in [0.5, 0.7, 1.0]:
        #     filename = ospj(args.sample_dir, '%06d_latent_psi_%.1f.jpg' % (step, psi))
        #     translate_using_latent(nets, args, src.x, y_trg_list, z_trg_list, psi, fname)
        #
        # fname = ospj(args.result_dir, 'latent.jpg')
        # print('Working on {}...'.format(fname))
        # utils.video_ref(nets_ema, args, src.x, ref.x, ref.y, fname)

    @torch.no_grad()
    def evaluate(self):
        args = self.args
        nets_ema = self.nets_ema
        resume_iter = args.resume_iter
        self._load_checkpoint(args.resume_iter)
        calculate_metrics(nets_ema, args, step=resume_iter, mode='latent')
        # calculate_metrics(nets_ema, args, step=resume_iter, mode='reference')


def compute_d_loss(nets, args, x1_source, x1_source_lm, x2_target, x2_target_lm, masks=None, loss_select = 'perceptual', embedder = None):

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
def compute_g_loss(nets, args, x1_source, x1_source_lm, x2_target, x2_target_lm,criterion_id, masks=None, loss_select = 'perceptual', embedder = None):

    # adversarial loss

    if args.id_embed:
        with torch.no_grad():
            s_fea = embedder(x1_source)
        s_trg = nets.mlp(s_fea)
    else:
        s_trg = nets.style_encoder(x1_source)
    # s_trg = nets.style_encoder(x1_source)

    x_fake = nets.generator(x2_target_lm, s_trg, masks=masks, loss_select=loss_select)
    out = nets.discriminator(x_fake, x2_target_lm)
    loss_adv = adv_loss(out, 1)

    # content reconstruction loss
    # s_pred = nets.style_encoder(x_fake)
    # loss_con = torch.mean(torch.abs(s_pred - s_trg))

    # cycle-consistency loss
    # x_fake_2 = nets.generator(x1_source_lm, s_pred, masks=masks)
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
        # x1_source = nn.functional.interpolate(x1_source[:, 0:1, :, :], size=(128, 128), mode='bilinear')
        x_fake = nn.functional.interpolate(x_fake[:, 0:1, :, :], size=(128, 128), mode='bilinear')
        x2_target = nn.functional.interpolate(x2_target[:, 0:1, :, :], size=(128, 128), mode='bilinear')
        # print(x1_source.size())

        criterion_id.eval()
        with torch.torch.no_grad():
            # _ = criterion_id(x1_source)
            # try:
            #     source_embs = criterion_id.feature
            # except:
            #     source_embs = criterion_id.module.feature
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



            # source_embs = criterion_id(x1_source[0:1])
            # target_embs = criterion_id(x2_target)
            # fake_embs = criterion_id(x_fake)
        # print(source_embs.size())
        # print(target_embs.size())
        # print(fake_embs.size())
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


def compute_d_loss_multi(nets, args, x1_1_source, x1_2_source, x1_3_source, x1_4_source, x2_target, x2_target_lm, masks=None, loss_select = 'perceptual'):

    # with real images
    x2_target.requires_grad_()
    out = nets.discriminator(x2_target, x2_target_lm)
    loss_real = adv_loss(out, 1)
    loss_reg = r1_reg(out, x2_target)

    # with fake images
    with torch.no_grad():
        s_trg = nets.style_encoder(x1_1_source)
        # print(s_trg)
        s_trg += nets.style_encoder(x1_2_source)
        s_trg += nets.style_encoder(x1_3_source)
        s_trg += nets.style_encoder(x1_4_source)
        # print(s_trg )
        # print(s_trg/4)
        # print(s_trg.size())
        # assert False
        # print(s_trg)
        s_trg_mean = s_trg/4
        # print(s_trg_mean)
        # print(s_trg_mean.size())
        # assert False


        x_fake = nets.generator(x2_target_lm, s_trg_mean, masks=masks, loss_select=loss_select )
    out = nets.discriminator(x_fake, x2_target_lm)
    loss_fake = adv_loss(out, 0)

    loss = loss_real + loss_fake + args.lambda_reg * loss_reg

    return loss, Munch(real=loss_real.item(),
                       fake=loss_fake.item(),
                       reg=loss_reg.item())

# def compute_g_loss(nets, args, x1_source, x2_target, x2_target_lm, arcface, conf, z_trgs=None, x_refs=None, masks=None):
def compute_g_loss_multi(nets, args, x1_1_source, x1_2_source, x1_3_source, x1_4_source, x2_target, x2_target_lm,criterion_id, masks=None, loss_select = 'perceptual'):

    # adversarial loss
    s_trg = nets.style_encoder(x1_1_source)
    s_trg += nets.style_encoder(x1_2_source)
    s_trg += nets.style_encoder(x1_3_source)
    s_trg += nets.style_encoder(x1_4_source)

    s_trg_mean = s_trg/4

    x_fake = nets.generator(x2_target_lm, s_trg_mean, masks=masks, loss_select=loss_select)
    out = nets.discriminator(x_fake, x2_target_lm)
    loss_adv = adv_loss(out, 1)

    # content reconstruction loss
    # s_pred = nets.style_encoder(x_fake)
    # loss_con = torch.mean(torch.abs(s_pred - s_trg))

    # cycle-consistency loss
    # x_fake_2 = nets.generator(x1_source_lm, s_pred, masks=masks)
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
        # print(param, param_test)
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

# import os
# print(os.getcwd())

# from arcface.config import get_config
#
#
# from arcface.Learner import face_learner
#
# def load_arcface():
#     conf = get_config(False)
#
#     # mtcnn = MTCNN()
#     # print('mtcnn loaded')
#
#     learner = face_learner(conf, True)
#
#     if torch.cuda.is_available():
#         learner.load_state(conf, 'mobilefacenet.pth', False, True)
#     else:
#         learner.load_state(conf, 'cpu_final.pth', True, True)
#     learner.model.eval()
#     # _, faces1 = mtcnn.align_multi(im1, conf.face_limit, conf.min_face_size)
#     # _, faces2 = mtcnn.align_multi(im2, conf.face_limit, conf.min_face_size)
#     # learner.extract_fea(conf, faces1[0], faces1[0], False)
#
#     return learner, conf

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
