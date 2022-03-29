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

from core.data_loader_lm_tran import InputFetcher_vox1
from core.data_loader_lm_tran import InputFetcher_test
import core.utils_lm_tran as utils
from metrics.eval import calculate_metrics
from tensorboardX import SummaryWriter


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
        self.writer = SummaryWriter('log/test')

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


        if self.args.vgg_encode:
            vgg_encode = network.vgg_feature(False, 0)


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
        if args.dataset == 'rafd':
            fetcher = InputFetcher_mpie(loaders.src, args.latent_dim, 'train')
            fetcher_val = InputFetcher_mpie(loaders.val, args.latent_dim, 'val')

            
        elif args.dataset == 'vox1':
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


            # fetch images and labels
            inputs = next(fetcher)
            # x_label, x2_label, x3_label, x4_label, x1_one_hot, x3_one_hot, x1_id, x3_id

            x1_label = inputs.x_label
            x2_label = inputs.x2_label
            x3_label = inputs.x3_label
            if  args.dataset == 'rafd':
                x4_label = inputs.x4_label

                param_x4 = x4_label[:, 0, :].unsqueeze(0)
                param_x4 = param_x4.view(-1, 136).float()
                param_x5 = None
            # elif args.dataset == 'vox1' and args.d_id:
            #
            #     x4_label = inputs.x4_label
            #     param_x4 = x4_label[:, 0, :].unsqueeze(0)
            #     param_x4 = param_x4.view(-1, 136).float()
            #
            #     x5_label = inputs.x5_label
            #     param_x5 = x5_label[:, 0, :].unsqueeze(0)
            #     param_x5 = param_x5.view(-1, 136).float()

            else:
                param_x4 = None
                param_x5 = None




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




            if args.dataset == 'vox1':

                one_hot_x1 = x1_one_hot[:, 0, :].unsqueeze(0)
                # one_hot_x1 = one_hot_x1.view(-1, 12606).float()
                one_hot_x1 = one_hot_x1.view(-1, 1251).float()


                one_hot_x3 = x3_one_hot[:, 0, :].unsqueeze(0)
                # one_hot_x3 = one_hot_x3.view(-1, 12606).float()
                one_hot_x3 = one_hot_x3.view(-1, 1251).float()
            elif args.dataset == 'rafd':
                one_hot_x1 = x1_one_hot[:, 0, :].unsqueeze(0)
                one_hot_x1 = one_hot_x1.view(-1, 67).float()

                one_hot_x3 = x3_one_hot[:, 0, :].unsqueeze(0)
                one_hot_x3 = one_hot_x3.view(-1, 67).float()




            if args.dataset == 'vox1':
                d_tran_loss, d_tran_losses = compute_d_tran_loss(
                    nets, args, param_x1,param_x2,param_x3,param_x4,one_hot_x1,one_hot_x3,x1_id, x3_id, masks=None, loss_select = args.loss, vgg_encode =vgg_encode)
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
                    nets, args, param_x1,param_x2,param_x3,param_x4,param_x5,one_hot_x1,one_hot_x3,x1_id, x3_id, masks=None, loss_select = args.loss, vgg_encode =vgg_encode)
                self._reset_grad()
                t_loss.backward()
                optims.linear_decoder.step()
                optims.lm_linear_encoder.step()


                moving_average(nets.linear_decoder, nets_ema.linear_decoder, beta=0.999)
                moving_average(nets.lm_linear_encoder, nets_ema.lm_linear_encoder, beta=0.999)

                
            elif args.dataset == 'rafd':
                # train the discriminator
                d_tran_loss, d_tran_losses = compute_d_tran_loss(
                    nets, args, param_x1,param_x2,param_x3,param_x4,one_hot_x1,one_hot_x3,x1_id, x3_id, masks=None, loss_select = args.loss, vgg_encode =vgg_encode)
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
                    nets, args, param_x1,param_x2,param_x3,param_x4, param_x5,one_hot_x1,one_hot_x3,x1_id, x3_id, masks=None, loss_select = args.loss, vgg_encode =vgg_encode)
                self._reset_grad()
                t_loss.backward()
                optims.linear_decoder.step()
                optims.lm_linear_encoder.step()


                moving_average(nets.linear_decoder, nets_ema.linear_decoder, beta=0.999)
                moving_average(nets.lm_linear_encoder, nets_ema.lm_linear_encoder, beta=0.999)





            # decay weight for diversity sensitive loss
            if args.lambda_ds > 0:
                args.lambda_ds -= (initial_lambda_ds / args.ds_iter)

            # print out log info
            if (i+1) % args.print_every == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
                log = "Elapsed time [%s], Iteration [%i/%i], " % (elapsed, i+1, args.total_iters)
                all_losses = dict()




                for loss, prefix in zip([d_tran_losses, t_losses],['D/', 'G/']):
                # for loss, prefix in zip([d_tran_losses, t_losses, c_losses],['D/', 'G/', 'C/']):

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
                utils.debug_image(nets_ema, args, inputs=inputs_val, step=i+1, vgg_encode =vgg_encode)

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

        vgg_encode = network.vgg_feature(False, 0)

        if args.dataset == 'mpie':
            fetcher_val = InputFetcher_mpie(loaders.src, args.latent_dim, 'test')

        elif args.dataset == '300vw':
            fetcher_val = InputFetcher_300vw(loaders.src, args.latent_dim, 'test')
        elif args.dataset == 'vox1':
            fetcher_val = InputFetcher_test(loaders.src, args.latent_dim, 'test')

        inputs = next(fetcher_val)

        # inputs = next(InputFetcher(loaders.src, args.latent_dim, 'test'))
        x1_label = inputs.x_label
        x2_label = inputs.x2_label
        x3_label = inputs.x3_label
        if args.dataset == 'mpie':
            x4_label = inputs.x4_label

            param_x4 = x4_label[:, 0, :].unsqueeze(0)
            param_x4 = param_x4.view(-1, 136).float()

        # x1_one_hot, x3_one_hot = inputs.x1_one_hot, inputs.x3_one_hot
        # x1_id, x3_id = inputs.x1_id, inputs.x3_id

        param_x1 = x1_label
        # param_x1 = x1_label[:, 0, :].unsqueeze(0)
        # param_x1 = param_x1.view(-1, 136).float()

        param_x2 = x2_label[:, 0, :].unsqueeze(0)
        param_x2 = param_x2.view(-1, 136).float()

        param_x3 = x3_label
        # param_x3 = x3_label[:, 0, :].unsqueeze(0)
        # param_x3 = param_x3.view(-1, 136).float()

        # one_hot_x1 = x1_one_hot[:, 0, :].unsqueeze(0)
        # one_hot_x1 = one_hot_x1.view(-1, 1000).float()
        #
        # one_hot_x3 = x3_one_hot[:, 0, :].unsqueeze(0)
        # one_hot_x3 = one_hot_x3.view(-1, 1000).float()

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

        if not os.path.exists(ospj(args.result_dir,'real')):
            os.makedirs(ospj(args.result_dir,'real'))
        if not os.path.exists(ospj(args.result_dir,'fake')):
            os.makedirs(ospj(args.result_dir,'fake'))
        if not os.path.exists(ospj(args.result_dir,'fake_rec')):
            os.makedirs(ospj(args.result_dir,'fake_rec'))

        fname_real = ospj(args.result_dir,'real', 'reconstruct.jpg')
        fname_fake = ospj(args.result_dir,'fake', 'reconstruct.jpg')
        fname_fake_rec = ospj(args.result_dir,'fake_rec', 'reconstruct.jpg')
        
        # fname = ospj(args.result_dir, 'reconstruct.jpg')
        # print('Working on {}...'.format(fname))
        # utils.translate_and_reconstruct_sample(nets_ema, args, src.x1, src.x1_c, src.x2, src.x2_lm, fname, conf, arcface)
        # utils.translate_and_reconstruct(nets_ema, args, src.x1, src.x_lm, src.x2, src.x2_lm, fname)

        utils.show_lm_test(nets_ema, args, param_x1, param_x2,param_x3, fname_real, fname_fake,fname_fake_rec,  vgg_encode =vgg_encode)
        # utils.show_lm(nets_ema, args, param_x1, param_x2, param_x3, param_x4, one_hot_x1, one_hot_x3, x1_id, x3_id, fname, vgg_encode =vgg_encode)
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
        # fname = ospj(args.resulout_2t_dir, 'latent.jpg')
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



    if args.dataset == 'rafd':
        # with real images
        x2_label.requires_grad_()

        out_2 = nets.linear_classfier(x2_label)

        loss_cls = classification_loss(out_2, x1_id)
        
    elif args.dataset == 'vox1':
        # with real images
        x2_label.requires_grad_()

        out_2 = nets.linear_classfier(x2_label)

        loss_cls = classification_loss(out_2, x1_id)

    loss =  args.lambda_cls * loss_cls

    return loss, Munch(loss_cls=loss_cls.item())

def compute_t_loss(nets, args, x1_label,x2_label,x3_label,x4_label,x5_label, one_hot_x1,one_hot_x3,x1_id, x3_id, masks=None, loss_select = 'perceptual', vgg_encode =None):
    if args.dataset == 'mpie':
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
        out_2 = nets.linear_classfier(x4_label)
    
        loss_adv = adv_loss(out_1, 1)
    
        loss_cls = classification_loss(out_2, x3_id)
    
        fake_fea_lm_2 = nets.lm_linear_encoder(fake_lm_2)
        real_fea_lm_2 = nets.lm_linear_encoder(x4_label)
    
        loss_con = torch.mean(torch.abs(fake_fea_lm_2 - real_fea_lm_2))
        
    elif args.dataset == 'vox1'  or args.dataset == 'rafd':

        # adversarial loss
        fea_id_1 = vgg_encode(x3_label)
        # fea_id_1 = nets.id_linear_encoder(fea_id_1)
        fea_lm_1 = nets.lm_linear_encoder(x2_label)

        # fea_id_1 = nets.id_linear_encoder(one_hot_x1)

        fake_lm_1 = nets.linear_decoder(fea_lm_1, fea_id_1)

        fea_lm_2 = nets.lm_linear_encoder(x2_label)

        fea_id_2 = vgg_encode(x1_label)
        # fea_id_2 = nets.id_linear_encoder(fea_id_2)
        # fea_id_2 = nets.id_linear_encoder(one_hot_x2)

        fake_lm_2 = nets.linear_decoder(fea_lm_2, fea_id_2)

        loss_cyc = np.zeros([])
        loss_cyc = torch.from_numpy(loss_cyc)
        loss_cyc = loss_cyc.type(torch.cuda.FloatTensor)

        for i in range(0, 68):
            loss_cyc += torch.mean(torch.abs(((fake_lm_2[:, 2 * i:2 * i + 1] - x2_label[:, 2 * i:2 * i + 1]) ** 2 + (
                    fake_lm_2[:, 2 * i + 1:2 * i + 2] - x2_label[:, 2 * i + 1:2 * i + 2]) ** 2) ** (0.5)))

        if args.dataset == 'rafd':
            loss_cyc_2 = np.zeros([])
            loss_cyc_2 = torch.from_numpy(loss_cyc_2)
            loss_cyc_2 = loss_cyc_2.type(torch.cuda.FloatTensor)

            for i in range(0, 68):
                loss_cyc_2 += torch.mean(
                    torch.abs(((fake_lm_1[:, 2 * i:2 * i + 1] - x4_label[:, 2 * i:2 * i + 1]) ** 2 + (
                            fake_lm_1[:, 2 * i + 1:2 * i + 2] - x4_label[:, 2 * i + 1:2 * i + 2]) ** 2) ** (0.5)))

        out_1 = nets.linear_discriminator(fake_lm_1)
        loss_adv = adv_loss(out_1, 1)



        # linear_eye_discriminator

        out_2 = nets.linear_classfier(fake_lm_1)
        loss_cls = classification_loss(out_2, x3_id)

        fake_fea_lm_2 = nets.lm_linear_encoder(fake_lm_1)
        fake_fea_lm_1 = nets.lm_linear_encoder(fake_lm_2)

        loss_con = torch.mean(torch.abs(fea_lm_1 - fake_fea_lm_1))

        # loss_cyc_eyes = np.zeros([])
        # loss_cyc_eyes = torch.from_numpy(loss_cyc_eyes)
        # loss_cyc_eyes = loss_cyc_eyes.type(torch.cuda.FloatTensor)
        #
        # for i in range(36, 48):
        #     loss_cyc_eyes += torch.mean(torch.abs(((fake_lm_1[:, 2 * i:2 * i + 1] - x2_label[:, 2 * i:2 * i + 1]) ** 2 + (
        #             fake_lm_1[:, 2 * i + 1:2 * i + 2] - x2_label[:, 2 * i + 1:2 * i + 2]) ** 2) ** (0.5)))



        loss_tp_contour_pos = np.zeros([])
        loss_tp_contour_pos = torch.from_numpy(loss_tp_contour_pos)
        loss_tp_contour_pos = loss_tp_contour_pos.type(torch.cuda.FloatTensor)

        loss_tp_contour_neg = np.zeros([])
        loss_tp_contour_neg = torch.from_numpy(loss_tp_contour_neg)
        loss_tp_contour_neg = loss_tp_contour_neg.type(torch.cuda.FloatTensor)

        loss_tp_eyebrow_pos = np.zeros([])
        loss_tp_eyebrow_pos = torch.from_numpy(loss_tp_eyebrow_pos)
        loss_tp_eyebrow_pos = loss_tp_eyebrow_pos.type(torch.cuda.FloatTensor)

        loss_tp_eyebrow_neg = np.zeros([])
        loss_tp_eyebrow_neg = torch.from_numpy(loss_tp_eyebrow_neg)
        loss_tp_eyebrow_neg = loss_tp_eyebrow_neg.type(torch.cuda.FloatTensor)

        loss_tp_nose_pos = np.zeros([])
        loss_tp_nose_pos = torch.from_numpy(loss_tp_nose_pos)
        loss_tp_nose_pos = loss_tp_nose_pos.type(torch.cuda.FloatTensor)

        loss_tp_nose_neg = np.zeros([])
        loss_tp_nose_neg = torch.from_numpy(loss_tp_nose_neg)
        loss_tp_nose_neg = loss_tp_nose_neg.type(torch.cuda.FloatTensor)

        loss_tp_eye_pos = np.zeros([])
        loss_tp_eye_pos = torch.from_numpy(loss_tp_eye_pos)
        loss_tp_eye_pos = loss_tp_eye_pos.type(torch.cuda.FloatTensor)

        loss_tp_eye_neg = np.zeros([])
        loss_tp_eye_neg = torch.from_numpy(loss_tp_eye_neg)
        loss_tp_eye_neg = loss_tp_eye_neg.type(torch.cuda.FloatTensor)

        loss_tp_mouth_pos = np.zeros([])
        loss_tp_mouth_pos = torch.from_numpy(loss_tp_mouth_pos)
        loss_tp_mouth_pos = loss_tp_mouth_pos.type(torch.cuda.FloatTensor)

        loss_tp_mouth_neg = np.zeros([])
        loss_tp_mouth_neg = torch.from_numpy(loss_tp_mouth_neg)
        loss_tp_mouth_neg = loss_tp_mouth_neg.type(torch.cuda.FloatTensor)



        for i in range(0, 17):
             loss_tp_contour_pos += torch.mean(torch.abs(((fake_lm_2[:, 2 * i:2 * i + 1] - x2_label[:, 2 * i:2 * i + 1]) ** 2 + (
                    fake_lm_2[:, 2 * i + 1:2 * i + 2] - x2_label[:, 2 * i + 1:2 * i + 2]) ** 2) ** (0.5)))

        for i in range(0, 17):
             loss_tp_contour_neg += torch.mean(torch.abs(((fake_lm_1[:, 2 * i:2 * i + 1] - x2_label[:, 2 * i:2 * i + 1]) ** 2 + (
                    fake_lm_1[:, 2 * i + 1:2 * i + 2] - x2_label[:, 2 * i + 1:2 * i + 2]) ** 2) ** (0.5)))

        for i in range(17, 27):
             loss_tp_eyebrow_pos += torch.mean(torch.abs(((fake_lm_2[:, 2 * i:2 * i + 1] - x2_label[:, 2 * i:2 * i + 1]) ** 2 + (
                    fake_lm_2[:, 2 * i + 1:2 * i + 2] - x2_label[:, 2 * i + 1:2 * i + 2]) ** 2) ** (0.5)))

        for i in range(17, 27):
             loss_tp_eyebrow_neg += torch.mean(torch.abs(((fake_lm_1[:, 2 * i:2 * i + 1] - x2_label[:, 2 * i:2 * i + 1]) ** 2 + (
                    fake_lm_1[:, 2 * i + 1:2 * i + 2] - x2_label[:, 2 * i + 1:2 * i + 2]) ** 2) ** (0.5)))

        for i in range(27, 36):
             loss_tp_nose_pos += torch.mean(torch.abs(((fake_lm_2[:, 2 * i:2 * i + 1] - x2_label[:, 2 * i:2 * i + 1]) ** 2 + (
                    fake_lm_2[:, 2 * i + 1:2 * i + 2] - x2_label[:, 2 * i + 1:2 * i + 2]) ** 2) ** (0.5)))

        for i in range(27, 36):
             loss_tp_nose_neg += torch.mean(torch.abs(((fake_lm_1[:, 2 * i:2 * i + 1] - x2_label[:, 2 * i:2 * i + 1]) ** 2 + (
                    fake_lm_1[:, 2 * i + 1:2 * i + 2] - x2_label[:, 2 * i + 1:2 * i + 2]) ** 2) ** (0.5)))

        for i in range(36, 48):
             loss_tp_eye_pos += torch.mean(torch.abs(((fake_lm_2[:, 2 * i:2 * i + 1] - x2_label[:, 2 * i:2 * i + 1]) ** 2 + (
                    fake_lm_2[:, 2 * i + 1:2 * i + 2] - x2_label[:, 2 * i + 1:2 * i + 2]) ** 2) ** (0.5)))

        for i in range(36, 48):
             loss_tp_eye_neg += torch.mean(torch.abs(((fake_lm_1[:, 2 * i:2 * i + 1] - x2_label[:, 2 * i:2 * i + 1]) ** 2 + (
                    fake_lm_1[:, 2 * i + 1:2 * i + 2] - x2_label[:, 2 * i + 1:2 * i + 2]) ** 2) ** (0.5)))


        for i in range(48, 68):
            loss_tp_mouth_pos += torch.mean(torch.abs(((fake_lm_2[:, 2 * i:2 * i + 1] - x2_label[:, 2 * i:2 * i + 1]) ** 2 + (
                    fake_lm_2[:, 2 * i + 1:2 * i + 2] - x2_label[:, 2 * i + 1:2 * i + 2]) ** 2) ** (0.5)))

        for i in range(48, 68):
            loss_tp_mouth_neg += torch.mean(torch.abs(((fake_lm_1[:, 2 * i:2 * i + 1] - x2_label[:, 2 * i:2 * i + 1]) ** 2 + (
                    fake_lm_1[:, 2 * i + 1:2 * i + 2] - x2_label[:, 2 * i + 1:2 * i + 2]) ** 2) ** (0.5)))



        loss_tp_eyebrow = 0.01 + loss_tp_eyebrow_pos - loss_tp_eyebrow_neg
        loss_tp_nose = 0.01 + loss_tp_nose_pos - loss_tp_nose_neg
        loss_tp_eye = 0.01 + loss_tp_eye_pos - loss_tp_eye_neg
        loss_tp_mouth = 0.01 + loss_tp_mouth_pos - loss_tp_mouth_neg

        # loss_tp_local = 0.005 + loss_tp_local_pos - loss_tp_local_neg

        loss_tp_contour = 0.2 + loss_tp_contour_pos - loss_tp_contour_neg

        # if loss_tp_local < 0:
        #     # loss_tp = loss_tp -loss_tp
        #     loss_tp_local = torch.abs(loss_tp_local)
        if loss_tp_eyebrow < 0:
            loss_tp_eyebrow = torch.abs(loss_tp_eyebrow)
        if loss_tp_nose < 0:
            loss_tp_nose = torch.abs(loss_tp_nose)
        if loss_tp_eye < 0:
            loss_tp_eye = torch.abs(loss_tp_eye)
        if loss_tp_mouth < 0:
            loss_tp_mouth = torch.abs(loss_tp_mouth)

        if loss_tp_contour < 0:
            loss_tp_contour = torch.abs(loss_tp_contour)


        loss_tp = loss_tp_eyebrow + loss_tp_nose + loss_tp_mouth + loss_tp_contour + loss_tp_eye





    if args.dataset == 'rafd':
        loss = args.lambda_d * loss_adv+ args.lambda_cyc * loss_cyc+ args.lambda_cyc2 * loss_cyc_2 + args.lambda_con * loss_con+ args.lambda_cls * loss_cls+ args.lambda_tp * loss_tp
        return loss, Munch(adv=loss_adv.item(),loss_cyc=loss_cyc.item(),loss_cyc_2=loss_cyc_2.item(),loss_cls=loss_cls.item(), loss_con=loss_con.item(),loss_tp=loss_tp.item())
    else:
        loss = args.lambda_d * loss_adv+ args.lambda_cyc * loss_cyc + args.lambda_con * loss_con+ args.lambda_cls * loss_cls+ args.lambda_tp * loss_tp
        return loss, Munch(adv=loss_adv.item(),loss_cyc=loss_cyc.item(),loss_cls=loss_cls.item(), loss_con=loss_con.item(),loss_tp=loss_tp.item())





def compute_d_tran_loss(nets, args, x1_label,x2_label,x3_label,x4_label,one_hot_x1,one_hot_x3,x1_id, x3_id,  masks=None, loss_select = 'perceptual', vgg_encode = None):

    if args.dataset == 'rafd':
        # with real images
        x2_label.requires_grad_()

        # out_1, out_2, out_3 = nets.discriminator_tran(x2_target_lm)
        out_1 = nets.linear_discriminator(x2_label)

        loss_real = adv_loss(out_1, 1)
        loss_reg = r1_reg(out_1, x2_label)

        # print(out_1, x4_label)
        # loss_cls = classification_loss(out_2, x3_id)

        with torch.no_grad():

            fea_lm_2 = nets.lm_linear_encoder(x2_label)
            # fea_id_2 = nets.id_linear_encoder(x3_label)
            # print(x1_label.size())
            # print(x2_label.size())
            # print(x3_label.size())
            # print(x4_label.size())
            # print(x3_label)
            fea_id_2 = vgg_encode(x3_label)
            # fea_id_2 = nets.id_linear_encoder(fea_id_2)

            fake_lm_2 = nets.linear_decoder(fea_lm_2, fea_id_2)

        out_2 = nets.linear_discriminator(fake_lm_2)

        loss_fake = adv_loss(out_2, 0)

        # loss = loss_real + loss_fake + args.lambda_reg * loss_reg + args.lambda_au_pose * loss_pose + args.lambda_au_aus * loss_aus
        loss = loss_real + loss_fake + args.lambda_reg * loss_reg  # + args.lambda_cls * loss_cls

        return loss, Munch(real=loss_real.item(),
                           fake=loss_fake.item(),
                           reg=loss_reg.item())

    elif args.dataset == 'vox1':
        # with real images
        x2_label.requires_grad_()

        # out_1, out_2, out_3 = nets.discriminator_tran(x2_target_lm)
        out_1 = nets.linear_discriminator(x2_label)

        loss_real = adv_loss(out_1, 1)
        loss_reg = r1_reg(out_1, x2_label)

        # print(out_1, x4_label)
        # loss_cls = classification_loss(out_2, x3_id)

        with torch.no_grad():
            

            fea_lm_2 = nets.lm_linear_encoder(x2_label)
            # fea_id_2 = nets.id_linear_encoder(x3_label)
            fea_id_2 = vgg_encode(x3_label)
            # fea_id_2 = nets.id_linear_encoder(fea_id_2)

            fake_lm_2 = nets.linear_decoder(fea_lm_2, fea_id_2)

        out_2 = nets.linear_discriminator(fake_lm_2)

        loss_fake = adv_loss(out_2, 0)

        # loss = loss_real + loss_fake + args.lambda_reg * loss_reg + args.lambda_au_pose * loss_pose + args.lambda_au_aus * loss_aus
        loss = loss_real + loss_fake + args.lambda_reg * loss_reg  # + args.lambda_cls * loss_cls

        return loss, Munch(real=loss_real.item(),
                           fake=loss_fake.item(),
                           reg=loss_reg.item())








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


