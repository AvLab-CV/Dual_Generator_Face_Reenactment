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

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.model_lm_talking_tran import build_model
from core.checkpoint import CheckpointIO
from core.data_loader_lm_tran import InputFetcher_mpie
from core.data_loader_lm_tran import InputFetcher_300vw
import core.utils_lm_tran as utils
from metrics.eval import calculate_metrics
from tensorboardX import SummaryWriter

from ms1m_ir50.model_irse import IR_50
from scipy import spatial

import network
from network.vgg import vgg_face, VGG_Activations
from torchvision.models import vgg19

import math
import numpy as np



class Solver(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.nets, self.nets_ema = build_model(args)
        # self.arcface, self.conf = load_arcface()
        # self.arcface = load_arcface_2()
        self.writer = SummaryWriter('log/test_45_lm_transformer_split_class_discriminator')
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




        args = self.args
        nets = self.nets
        nets_ema = self.nets_ema
	
        optims = self.optims
        for net in nets.keys():
            if net == 'linear_classfier':

                optims[net] = torch.optim.Adam(
                    params=nets[net].parameters(),
                    lr=args.lr2,
                    betas=[args.beta1, args.beta2],
                    weight_decay=args.weight_decay)

        print(optims)
        print(self.nets.keys())
        # assert False

        # fetch random validation images for debugging
        if args.dataset == 'mpie':
            fetcher = InputFetcher_mpie(loaders.src, args.latent_dim, 'train')
            fetcher_val = InputFetcher_mpie(loaders.val, args.latent_dim, 'val')

        elif args.dataset == '300vw':
            fetcher = InputFetcher_300vw(loaders.src, args.latent_dim, 'train')
            fetcher_val = InputFetcher_300vw(loaders.val, args.latent_dim, 'val')
        inputs_val = next(fetcher_val)

        # resume training if necessary
        if args.resume_iter > 0:
            self._load_checkpoint(args.resume_iter)

        # remember the initial value of ds weight
        initial_lambda_ds = args.lambda_ds

        print('Start training...')
        start_time = time.time()
        for i in range(args.resume_iter, args.total_iters):
            if (i+1) % args.decay_every == 0:

                # print('54555')
                times = (i+1) /args.decay_every
                # print(args.lr*0.1**int(times))
                optims = Munch()
                for net in nets.keys():
                    if net == 'fan' :
                        continue
                    optims[net] = torch.optim.Adam(
                        params=nets[net].parameters(),
                        lr=args.lr*0.1**int(times),
                        betas=[args.beta1, args.beta2],
                        weight_decay=args.weight_decay)
                        
                # optims = torch.optim.Adam(
                #             params=self.nets[net].parameters(),
                #             lr=args.lr*0.1**int(times),
                #             betas=[args.beta1, args.beta2],
                #             weight_decay=args.weight_decay)

            # fetch images and labels
            inputs = next(fetcher)
            # x_label, x2_label, x3_label, x4_label, x1_one_hot, x3_one_hot, x1_id, x3_id

            x1_label = inputs.x_label
            x2_label = inputs.x2_label
            x3_label = inputs.x3_label
            if args.dataset == 'mpie':
                x4_label = inputs.x4_label

                param_x4 = x4_label[:, 0, :].unsqueeze(0)
                param_x4 = param_x4.view(-1, 136).float()




            x1_one_hot, x3_one_hot = inputs.x1_one_hot, inputs.x3_one_hot
            x1_id, x3_id = inputs.x1_id, inputs.x3_id

            param_x1 = x1_label[:, 0, :].unsqueeze(0)
            param_x1 = param_x1.view(-1, 136).float()


            param_x2 = x2_label[:, 0, :].unsqueeze(0)
            param_x2 = param_x2.view(-1, 136).float()

            param_x3 = x3_label[:, 0, :].unsqueeze(0)
            param_x3 = param_x3.view(-1, 136).float()










            one_hot_x1 = x1_one_hot[:, 0, :].unsqueeze(0)
            one_hot_x1 = one_hot_x1.view(-1, 150).float()


            one_hot_x3 = x3_one_hot[:, 0, :].unsqueeze(0)
            one_hot_x3 = one_hot_x3.view(-1, 150).float()

            # print(param_x1.shape)
            # print(one_hot_x1.shape)
            # assert False



            # masks = nets.fan.get_heatmap(x_real) if args.w_hpf > 0 else None

            # linear_decoder = Linear_decoder()
            # id_linear_encoder = Id_linear_encoder()
            # lm_linear_encoder = Lm_linear_encoder()
            # linear_discriminator = Linear_discriminator()

            if args.dataset == '300vw':
                print('300vw')

            elif args.dataset == 'mpie':
                # train the discriminator
                d_tran_loss, d_tran_losses = compute_d_tran_loss(
                    nets, args, param_x1,param_x2,param_x3,param_x4,one_hot_x1,one_hot_x3,x1_id, x3_id, masks=None, loss_select = args.loss)
                self._reset_grad()
                d_tran_loss.backward()
                optims.linear_discriminator.step()
                moving_average(nets.linear_discriminator, nets_ema.linear_discriminator, beta=0.999)

                # train the classfier
                c_loss, c_losses = compute_c_loss(
                    nets, args, param_x1,param_x2,param_x3,param_x4,one_hot_x1,one_hot_x3,x1_id, x3_id, masks=None, loss_select = args.loss)
                self._reset_grad()
                c_loss.backward()
                optims.linear_classfier.step()
                moving_average(nets.linear_classfier, nets_ema.linear_classfier, beta=0.999)

                # train the transformer
                t_loss, t_losses = compute_t_loss(
                    nets, args, param_x1,param_x2,param_x3,param_x4,one_hot_x1,one_hot_x3,x1_id, x3_id, masks=None, loss_select = args.loss)
                self._reset_grad()
                t_loss.backward()
                optims.linear_decoder.step()
                optims.lm_linear_encoder.step()
                optims.id_linear_encoder.step()

                moving_average(nets.linear_decoder, nets_ema.linear_decoder, beta=0.999)
                moving_average(nets.lm_linear_encoder, nets_ema.lm_linear_encoder, beta=0.999)
                moving_average(nets.id_linear_encoder, nets_ema.id_linear_encoder, beta=0.999)





            # # train the discriminator
            # d_loss, d_losses = compute_d_loss(
            #     nets, args, x1_source,x1_source_lm, x2_target, x2_target_lm, masks=None, loss_select = args.loss)
            # self._reset_grad()
            # d_loss.backward()
            # optims.discriminator.step()
            #
            #
            #
            # # train the generator
            # g_loss, g_losses = compute_g_loss(
            #     nets, args, x1_source, x1_source_lm, x2_target, x2_target_lm, criterion_id, masks=None, loss_select = args.loss)
            # self._reset_grad()
            # g_loss.backward()
            #
            # if args.transformer:
            #     optims.lm_encoder.step()
            #     optims.lm_transformer.step()
            #     optims.lm_decoder.step()
            #     optims.style_encoder.step()
            # else:
            #     optims.generator.step()
            #     optims.style_encoder.step()
            #
            # if args.transformer:
            #     moving_average(nets.lm_encoder, nets_ema.lm_encoder, beta=0.999)
            #     moving_average(nets.lm_transformer, nets_ema.lm_transformer, beta=0.999)
            #     moving_average(nets.lm_decoder, nets_ema.lm_decoder, beta=0.999)
            #     moving_average(nets.style_encoder, nets_ema.style_encoder, beta=0.999)
            #
            # else:
            #     # compute moving average of network parameters
            #     moving_average(nets.generator, nets_ema.generator, beta=0.999)
            #     # moving_average(nets.mapping_network, nets_ema.mapping_network, beta=0.999)
            #     moving_average(nets.style_encoder, nets_ema.style_encoder, beta=0.999)



            # decay weight for diversity sensitive loss
            if args.lambda_ds > 0:
                args.lambda_ds -= (initial_lambda_ds / args.ds_iter)

            # print out log info
            if (i+1) % args.print_every == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
                log = "Elapsed time [%s], Iteration [%i/%i], " % (elapsed, i+1, args.total_iters)
                all_losses = dict()
                # for loss, prefix in zip([d_losses, g_losses],['D/', 'G/']):
                for loss, prefix in zip([d_tran_losses, t_losses, c_losses],['D/', 'G/', 'C/']):

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
                utils.debug_image(nets_ema, args, inputs=inputs_val, step=i+1)

            # save model checkpoints
            if (i+1) % args.save_every == 0:
                self._save_checkpoint(step=i+1)

            # compute FID and LPIPS if necessary
            if (i+1) % args.eval_every == 0:
                calculate_metrics(nets_ema, args, i+1, mode='latent')
                calculate_metrics(nets_ema, args, i+1, mode='reference')
        self.writer.close()

    @torch.no_grad()
    def sample(self, loaders):



        args = self.args
        nets_ema = self.nets_ema
        print(nets_ema)
        os.makedirs(args.result_dir, exist_ok=True)
        self._load_checkpoint(args.resume_iter)

        if args.dataset == 'mpie':
            fetcher_val = InputFetcher_mpie(loaders.src, args.latent_dim, 'test')

        elif args.dataset == '300vw':
            fetcher_val = InputFetcher_300vw(loaders.src, args.latent_dim, 'test')
        inputs = next(fetcher_val)

        # inputs = next(InputFetcher(loaders.src, args.latent_dim, 'test'))
        x1_label = inputs.x_label
        x2_label = inputs.x2_label
        x3_label = inputs.x3_label
        if args.dataset == 'mpie':
            x4_label = inputs.x4_label

            param_x4 = x4_label[:, 0, :].unsqueeze(0)
            param_x4 = param_x4.view(-1, 136).float()

        x1_one_hot, x3_one_hot = inputs.x1_one_hot, inputs.x3_one_hot
        x1_id, x3_id = inputs.x1_id, inputs.x3_id

        param_x1 = x1_label[:, 0, :].unsqueeze(0)
        param_x1 = param_x1.view(-1, 136).float()

        param_x2 = x2_label[:, 0, :].unsqueeze(0)
        param_x2 = param_x2.view(-1, 136).float()

        param_x3 = x3_label[:, 0, :].unsqueeze(0)
        param_x3 = param_x3.view(-1, 136).float()

        one_hot_x1 = x1_one_hot[:, 0, :].unsqueeze(0)
        one_hot_x1 = one_hot_x1.view(-1, 150).float()

        one_hot_x3 = x3_one_hot[:, 0, :].unsqueeze(0)
        one_hot_x3 = one_hot_x3.view(-1, 150).float()

        # x1_label = inputs.x_label
        # x2_label = inputs.x2_label
        # x3_label = inputs.x3_label
        # x4_label = inputs.x4_label
        #
        # x1_one_hot, x3_one_hot = inputs.x1_one_hot, inputs.x3_one_hot
        # x1_id, x3_id = inputs.x1_id, inputs.x3_id
        #
        # param_x1 = x1_label[:, 0, :].unsqueeze(0)
        # param_x1 = param_x1.view(-1, 136).float()
        #
        # param_x2 = x2_label[:, 0, :].unsqueeze(0)
        # param_x2 = param_x2.view(-1, 136).float()
        #
        # param_x3 = x3_label[:, 0, :].unsqueeze(0)
        # param_x3 = param_x3.view(-1, 136).float()
        #
        # param_x4 = x4_label[:, 0, :].unsqueeze(0)
        # param_x4 = param_x4.view(-1, 136).float()
        #
        # one_hot_x1 = x1_one_hot[:, 0, :].unsqueeze(0)
        # one_hot_x1 = one_hot_x1.view(-1, 150).float()
        #
        # one_hot_x3 = x3_one_hot[:, 0, :].unsqueeze(0)
        # one_hot_x3 = one_hot_x3.view(-1, 150).float()


        fname = ospj(args.result_dir, 'reconstruct.jpg')
        print('Working on {}...'.format(fname))
        # utils.translate_and_reconstruct_sample(nets_ema, args, src.x1, src.x1_c, src.x2, src.x2_lm, fname, conf, arcface)
        # utils.translate_and_reconstruct(nets_ema, args, src.x1, src.x_lm, src.x2, src.x2_lm, fname)
        utils.show_lm(nets_ema, args, param_x1, param_x2, param_x3, param_x4, one_hot_x1, one_hot_x3, x1_id, x3_id, fname)
        # utils.show_lm(nets_ema, args, src.x1, src.x_lm, src.x2, src.x2_lm, fname, param_x1, param_x2)
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
        calculate_metrics(nets_ema, args, step=resume_iter, mode='reference')

def compute_c_loss(nets, args, x1_label,x2_label,x3_label,x4_label,one_hot_x1,one_hot_x3,x1_id, x3_id,  masks=None, loss_select = 'perceptual'):

    # with real images
    x4_label.requires_grad_()

    # out_1, out_2, out_3 = nets.discriminator_tran(x2_target_lm)
    out_2 = nets.linear_classfier(x4_label)


    # print(out_1, x4_label)
    loss_cls = classification_loss(out_2, x3_id)


    # # with fake images
    # with torch.no_grad():
    #
    #     fea_lm_2 = nets.lm_linear_encoder(x2_label)
    #     fea_id_2 = nets.id_linear_encoder(x3_label)
    #
    #     fake_lm_2 = nets.linear_decoder(fea_lm_2, fea_id_2)
    #
    #
    #
    #
    # out_2 = nets.linear_classfier(fake_lm_2)


    # print(out_1, x4_label)
    # loss_cls_fake = classification_loss(out_2, x3_id)

    # loss_cls = loss_cls_real + loss_cls_fake
    # loss = loss_real + loss_fake + args.lambda_reg * loss_reg + args.lambda_au_pose * loss_pose + args.lambda_au_aus * loss_aus
    loss =  args.lambda_cls * loss_cls

    return loss, Munch(loss_cls=loss_cls.item())

def compute_t_loss(nets, args, x1_label,x2_label,x3_label,x4_label,one_hot_x1,one_hot_x3,x1_id, x3_id, masks=None, loss_select = 'perceptual'):

    # adversarial loss

    fea_id_1 = nets.id_linear_encoder(x1_label)
    fea_lm_1 = nets.lm_linear_encoder(x2_label)

    # fea_id_1 = nets.id_linear_encoder(one_hot_x1)

    fake_lm_1 = nets.linear_decoder(fea_lm_1, fea_id_1)


    loss_lm = np.zeros([])
    loss_lm =torch.from_numpy(loss_lm)
    loss_lm = loss_lm.type(torch.cuda.FloatTensor)

    # print(fake_lm_1.shape)
    # print(x2_label.shape)
    for i in range(0, 68):
        loss_lm += torch.mean(torch.abs(((fake_lm_1[:, 2*i:2*i+1] - x2_label[:, 2*i:2*i+1])**2+(fake_lm_1[:, 2*i+1:2*i+2] - x2_label[:, 2*i+1:2*i+2])**2)**(0.5)))
    # loss_lm = loss_lm / 68
    # loss_lm = torch.mean(torch.abs(fake_lm_1 - x2_label))


    fea_lm_2 = nets.lm_linear_encoder(x2_label)
    fea_id_2 = nets.id_linear_encoder(x3_label)
    # fea_id_2 = nets.id_linear_encoder(one_hot_x2)

    fake_lm_2 = nets.linear_decoder(fea_lm_2, fea_id_2)

    loss_lm_2 = np.zeros([])
    loss_lm_2 =torch.from_numpy(loss_lm_2)
    loss_lm_2 = loss_lm_2.type(torch.cuda.FloatTensor)

    for i in range(0, 68):
    #     loss_lm_2 += torch.abs(((fake_lm_2[:, 2 * i:2 * i + 1] - x4_label[:, 2 * i:2 * i + 1]) ** 2 + (
    #                 fake_lm_2[:, 2 * i + 1:2 * i + 2] - x4_label[:, 2 * i + 1:2 * i + 2]) ** 2) ** (0.5))

        loss_lm_2 += torch.mean(torch.abs(((fake_lm_2[:, 2*i:2*i+1] - x4_label[:, 2*i:2*i+1])**2+(fake_lm_2[:, 2*i+1:2*i+2] - x4_label[:, 2*i+1:2*i+2])**2)**(0.5)))
    # loss_lm_2 = torch.mean(torch.abs(fake_lm_2 - x4_label))
    # loss_lm_2 = loss_lm_2/68

    out_1 = nets.linear_discriminator(fake_lm_2)
    out_2 = nets.linear_classfier(fake_lm_2)

    loss_adv = adv_loss(out_1, 1)

    loss_cls = classification_loss(out_2, x3_id)

    fake_fea_lm_2 = nets.lm_linear_encoder(fake_lm_2)
    real_fea_lm_2 = nets.lm_linear_encoder(x4_label)

    loss_con = torch.mean(torch.abs(fake_fea_lm_2 - real_fea_lm_2))




    loss =  args.lambda_d*loss_adv + args.lambda_cls * loss_cls+ args.lambda_lm * loss_lm+ args.lambda_lm * loss_lm_2 + args.lambda_con * loss_con

    return loss, Munch(adv=loss_adv.item(),
                       loss_cls=loss_cls.item(),loss_lm=loss_lm.item(),loss_lm_2=loss_lm_2.item(),loss_con=loss_con.item())




# linear_decoder = Linear_decoder()
# id_linear_encoder = Id_linear_encoder()
# lm_linear_encoder = Lm_linear_encoder()
# linear_discriminator = Linear_discriminator()

def compute_d_tran_loss(nets, args, x1_label,x2_label,x3_label,x4_label,one_hot_x1,one_hot_x3,x1_id, x3_id,  masks=None, loss_select = 'perceptual'):

    # with real images
    x4_label.requires_grad_()

    # out_1, out_2, out_3 = nets.discriminator_tran(x2_target_lm)
    out_1 = nets.linear_discriminator(x4_label)

    loss_real = adv_loss(out_1, 1)
    loss_reg = r1_reg(out_1, x4_label)

    # print(out_1, x4_label)
    # loss_cls = classification_loss(out_2, x3_id)



    with torch.no_grad():

        fea_lm_2 = nets.lm_linear_encoder(x2_label)
        fea_id_2 = nets.id_linear_encoder(x3_label)

        fake_lm_2 = nets.linear_decoder(fea_lm_2, fea_id_2)




    out_1 = nets.linear_discriminator(fake_lm_2)


    loss_fake = adv_loss(out_1, 0)




    # loss = loss_real + loss_fake + args.lambda_reg * loss_reg + args.lambda_au_pose * loss_pose + args.lambda_au_aus * loss_aus
    loss =  args.lambda_d*loss_real +  args.lambda_d*loss_fake + args.lambda_reg * loss_reg #+ args.lambda_cls * loss_cls

    return loss, Munch(real=loss_real.item(),
                       fake=loss_fake.item(),
                       reg=loss_reg.item())
    # return loss, Munch(real=loss_real.item(),
    #                    fake=loss_fake.item(),
    #                    reg=loss_reg.item(),
    #                    loss_pose_real=loss_pose_real.item(),
    #                    loss_aus_real=loss_aus_real.item(),
    #                    loss_pose_fake=loss_pose_fake.item(),
    #                    loss_aus_fake=loss_aus_fake.item())

def compute_e_loss(nets, args, x1_source, x1_source_lm, x2_target, x2_target_lm, AUR_x1, AUR_x2, x2_label,
                        masks=None, loss_select='perceptual'):
    # with real images
    x2_target_lm.requires_grad_()

    # out_1, out_2, out_3 = nets.discriminator_tran(x2_target_lm)
    out_2, out_3 = nets.estimator_tran(x2_target_lm)


    param_x1 = out_2.view(-1, 3)

    real_AUR_pose = param_x1.view(param_x1.size(0), param_x1.size(1), 1, 1).expand(
        param_x1.size(0), param_x1.size(1), args.img_size, args.img_size)

    param_x2 = out_3.view(-1, 17)
    real_AUR_aus = param_x2.view(param_x2.size(0), param_x2.size(1), 1, 1).expand(
        param_x2.size(0), param_x2.size(1), args.img_size, args.img_size)


    # with fake images
    with torch.no_grad():
        fake_lm = nets.transformer(x1_source, AUR_x2)

    # out_1, out_2, out_3 = nets.discriminator_tran(fake_lm)
    out_2, out_3 = nets.estimator_tran(fake_lm)

    param_x3 = out_2.view(-1, 3)

    fake_AUR_pose = param_x3.view(param_x3.size(0), param_x3.size(1), 1, 1).expand(
        param_x3.size(0), param_x3.size(1), args.img_size, args.img_size)

    param_x4 = out_3.view(-1, 17)
    fake_AUR_aus = param_x4.view(param_x4.size(0), param_x4.size(1), 1, 1).expand(
        param_x4.size(0), param_x4.size(1), args.img_size, args.img_size)

    print('target pose')
    print((((x2_label[:, 0:3]) * math.pi - (math.pi / 2)) * 360) / (2 * math.pi))
    print('real pose')
    print((((param_x1) * math.pi - (math.pi / 2)) * 360) / (2 * math.pi))
    print('fake pose')
    print((((param_x3) * math.pi - (math.pi / 2)) * 360) / (2 * math.pi))

    print('target aus')
    print(x2_label[:, 3:] * 5)
    print('real aus')
    print(param_x2 * 5)
    print('fake aus')
    print(param_x4 * 5)

    loss_pose_real = torch.mean(F.mse_loss(AUR_x2[:, 0:3, :, :], real_AUR_pose))
    loss_aus_real = torch.mean(F.mse_loss(AUR_x2[:, 3:, :, :], real_AUR_aus))

    loss_pose_fake = torch.mean(F.mse_loss(AUR_x2[:, 0:3, :, :], fake_AUR_pose))
    loss_aus_fake = torch.mean(F.mse_loss(AUR_x2[:, 3:, :, :], fake_AUR_aus))

    loss_pose = loss_pose_real + loss_pose_fake
    loss_aus = loss_aus_real + loss_aus_fake

    loss = args.lambda_au_pose * loss_pose + args.lambda_au_aus * loss_aus

    return loss, Munch(loss_pose_real=loss_pose_real.item(),
                       loss_aus_real=loss_aus_real.item(),
                       loss_pose_fake=loss_pose_fake.item(),
                       loss_aus_fake=loss_aus_fake.item())




def compute_d_loss(nets, args, x1_source, x1_source_lm, x2_target, x2_target_lm, masks=None, loss_select = 'perceptual'):

    # with real images
    x2_target.requires_grad_()
    if args.transformer:
        out = nets.discriminator(x2_target)
    else:
        out = nets.discriminator(x2_target, x2_target_lm)
    # out = nets.discriminator(x2_target, x2_target_lm)
    loss_real = adv_loss(out, 1)
    loss_reg = r1_reg(out, x2_target)

    # with fake images
    with torch.no_grad():
        s_trg = nets.style_encoder(x1_source)

        if args.transformer:
            target_lm_fea = nets.lm_encoder(x2_target_lm, s_trg, masks=masks, loss_select=loss_select)
            source_lm_fea = nets.lm_encoder(x1_source_lm, s_trg, masks=masks, loss_select=loss_select)

            final_lm_fea = nets.lm_transformer(target_lm_fea, source_lm_fea, masks=masks, loss_select=loss_select)

            x_fake = nets.lm_decoder(final_lm_fea, s_trg, masks=masks, loss_select=loss_select)

        else:
            x_fake = nets.generator(x2_target_lm, s_trg, masks=masks, loss_select=loss_select)

    if args.transformer:
        out = nets.discriminator(x_fake)
    else:
        out = nets.discriminator(x_fake, x2_target_lm)
    # out = nets.discriminator(x_fake, x2_target_lm)
    loss_fake = adv_loss(out, 0)

    loss = loss_real + loss_fake + args.lambda_reg * loss_reg

    return loss, Munch(real=loss_real.item(),
                       fake=loss_fake.item(),
                       reg=loss_reg.item())



# def compute_g_loss(nets, args, x1_source, x2_target, x2_target_lm, arcface, conf, z_trgs=None, x_refs=None, masks=None):
def compute_g_loss(nets, args, x1_source, x1_source_lm, x2_target, x2_target_lm,criterion_id, masks=None, loss_select = 'perceptual'):

    # adversarial loss
    s_trg = nets.style_encoder(x1_source)
    if args.transformer:
        target_lm_fea = nets.lm_encoder(x2_target_lm, s_trg, masks=masks, loss_select=loss_select)
        source_lm_fea = nets.lm_encoder(x1_source_lm, s_trg, masks=masks, loss_select=loss_select)

        final_lm_fea = nets.lm_transformer(target_lm_fea, source_lm_fea, masks=masks, loss_select=loss_select)

        x_fake = nets.lm_decoder(final_lm_fea, s_trg, masks=masks, loss_select=loss_select)

    else:
        x_fake = nets.generator(x2_target_lm, s_trg, masks=masks, loss_select=loss_select)

    if args.transformer:
        out = nets.discriminator(x_fake)
    else:
        out = nets.discriminator(x_fake, x2_target_lm)

    loss_adv = adv_loss(out, 1)

    # content reconstruction loss
    # s_pred = nets.style_encoder(x_fake)
    # loss_con = torch.mean(torch.abs(s_pred - s_trg))

    # cycle-consistency loss
    # x_fake_2 = nets.generator(x1_source_lm, s_pred, masks=masks)
    if not args.transformer:
        loss_cyc_1 = torch.mean(torch.abs(x_fakqe - x2_target))
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

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        if args.transformer:
            output = cos(source_embs, fake_embs)
            loss_id = torch.mean(1 - output)
        else:
            output = cos(target_embs, fake_embs)
            loss_id = torch.mean(1 - output)
        # output2 = cos(target_embs, source_embs)
        # loss_id2 = torch.mean(1 - output2)
        # print('loss_id: {}'.format(loss_id))
        # print('loss_id2: {}'.format(loss_id2))


    elif args.loss == 'perceptual':
        if args.transformer:
            loss_id = criterion_id(x1_source, x2_target)
        else:
            loss_id = criterion_id(x_fake, x2_target)



    # loss = loss_adv  + args.lambda_cyc * loss_cyc_1 + args.lambda_cyc * loss_cyc_2 + args.lambda_con * loss_con + args.lambda_id * loss_id_1 + args.lambda_id * loss_id_2

    # loss = loss_adv + args.lambda_cyc * loss_cyc_1 + args.lambda_id * loss_id
    # return loss, Munch(adv=loss_adv.item(),
    #                    cyc_1=loss_cyc_1.item(),
    #                    cyc_2=loss_cyc_2.item(),
    #                    con=loss_con.item(),
    #                    id_1=loss_id_1.item(),
    #                    id_2=loss_id_2.item())
    if args.transformer:
        loss = loss_adv + args.lambda_id * loss_id
        return loss, Munch(adv=loss_adv.item(),
                       id=loss_id.item())
    else:
        loss = loss_adv + args.lambda_cyc * loss_cyc_1 + args.lambda_id * loss_id
        return loss, Munch(adv=loss_adv.item(),
                       cyc_1=loss_cyc_1.item(),
                       id=loss_id.item())





def moving_average(model, model_test, beta=0.999):
    for param, param_test in zip(model.parameters(), model_test.parameters()):
        param_test.data = torch.lerp(param.data, param_test.data, beta)


def adv_loss(logits, target):
    assert target in [1, 0]
    # print(logits.dtype)
    #
    # print(type(logits))

    # print(logits.size())


    targets = torch.full_like(logits, fill_value=target)
    # print(targets.dtype)
    # print(type(targets))
    # print(targets.size())
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss

def classification_loss(logit, target):
    """Compute binary or softmax cross entropy loss."""
    # logit = logit.long()
    # device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    # noise = torch.randn(N, 100, device=device)
    # print(target.squeeze(0).shape)
    # print(target.view(-1).shape)
    # target = torch.tensor([0,1,2,3])
    # print(target)
    target = target.view(-1)
    # target = target.long()
    # target.astype(torch.int64)
    # print(logit.shape)

    # print()
    return F.cross_entropy(logit, target)


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
