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

from core.model_lm_talking import build_model
from core.checkpoint import CheckpointIO
from core.data_loader_lm import InputFetcher
import core.utils_lm as utils
from metrics.eval import calculate_metrics
from tensorboardX import SummaryWriter

from ms1m_ir50.model_irse import IR_50
from scipy import spatial




class Solver(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.nets, self.nets_ema = build_model(args)
        # self.arcface, self.conf = load_arcface()
        # self.arcface = load_arcface_2()
        self.writer = SummaryWriter('log/test33_arcface_adv_cycle_id_add_lm_in_D')
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
        BACKBONE_RESUME_ROOT = 'D:/face-recognition/stargan-v2-master/ms1m_ir50/backbone_ir50_ms1m_epoch120.pth'

        INPUT_SIZE = [112, 112]
        arcface = IR_50(INPUT_SIZE)

        if os.path.isfile(BACKBONE_RESUME_ROOT):
            arcface.load_state_dict(torch.load(BACKBONE_RESUME_ROOT))
            print("Loading Backbone Checkpoint '{}'".format(BACKBONE_RESUME_ROOT))

        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        arcface = arcface.to(DEVICE)



        args = self.args
        nets = self.nets
        # conf = self.conf
        # arcface = self.arcface
        nets_ema = self.nets_ema
        optims = self.optims

        # fetch random validation images for debugging
        fetcher = InputFetcher(loaders.src, args.latent_dim, 'train')
        fetcher_val = InputFetcher(loaders.val, args.latent_dim, 'val')
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

            x1_source, x1_source_lm = inputs.x1, inputs.x_lm
            x2_target, x2_target_lm = inputs.x2, inputs.x2_lm

            # x_source_4_channel, x2_target, x2_target_lm = inputs.x1_c, inputs.x2, inputs.x2_lm
            # x_source_4_channel = nn.functional.interpolate(x_source_4_channel[:, :, :, :], size=(128, 128), mode='bilinear')



            # x_real, y_org = inputs.x_src, inputs.y_src
            # x_ref, x_ref2, y_trg = inputs.x_ref, inputs.x_ref2, inputs.y_ref
            # z_trg, z_trg2 = inputs.z_trg, inputs.z_trg2

            # masks = nets.fan.get_heatmap(x_real) if args.w_hpf > 0 else None

            # train the discriminator
            d_loss, d_losses = compute_d_loss(
                nets, args, x1_source, x1_source_lm, x2_target, x2_target_lm, z_trg=None, masks=None)
            self._reset_grad()
            d_loss.backward()
            optims.discriminator.step()

            # d_loss, d_losses_ref = compute_d_loss(
            #     nets, args, x_real, y_org, y_trg, x_ref=x_ref, masks=masks)
            # self._reset_grad()
            # d_loss.backward()
            # optims.discriminator.step()

            # train the generator
            # g_loss, g_losses = compute_g_loss(
            #     nets, args, x1_source, x2_target, x2_target_lm, arcface, conf, z_trgs=None, masks=None)
            g_loss, g_losses = compute_g_loss(
                nets, args, x1_source, x1_source_lm, x2_target, x2_target_lm, arcface, z_trgs=None, masks=None)
            self._reset_grad()
            g_loss.backward()
            optims.generator.step()
            optims.style_encoder.step()

            # g_loss, g_losses_ref = compute_g_loss(
            #     nets, args, x_real, y_org, y_trg, arcface, conf, x_refs=[x_ref, x_ref2], masks=masks)
            # self._reset_grad()
            # g_loss.backward()
            # optims.generator.step()

            # compute moving average of network parameters
            moving_average(nets.generator, nets_ema.generator, beta=0.999)
            # moving_average(nets.mapping_network, nets_ema.mapping_network, beta=0.999)
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

        # conf = self.conf
        arcface = self.arcface

        # loss_id, _ = arcface.extract_fea(args, conf, x1_source, x_fake, False)



        args = self.args
        nets_ema = self.nets_ema
        os.makedirs(args.result_dir, exist_ok=True)
        self._load_checkpoint(args.resume_iter)

        src = next(InputFetcher(loaders.src, args.latent_dim, 'test'))
        # ref = next(InputFetcher(loaders.ref, None, args.latent_dim, 'test'))

        fname = ospj(args.result_dir, 'reconstruct.jpg')
        print('Working on {}...'.format(fname))
        utils.translate_and_reconstruct_sample(nets_ema, args, src.x1, src.x1_c, src.x2, src.x2_lm, fname, conf, arcface)
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


def compute_d_loss(nets, args, x1_source,x1_source_lm, x2_target, x2_target_lm, z_trg=None, x_ref=None, masks=None):

    # with real images
    x2_target.requires_grad_()
    out = nets.discriminator(x2_target, x2_target_lm)
    loss_real = adv_loss(out, 1)
    loss_reg = r1_reg(out, x2_target)

    # with fake images
    with torch.no_grad():
        s_trg = nets.style_encoder(x1_source)

        x_fake = nets.generator(x2_target_lm, s_trg, masks=masks)
    out = nets.discriminator(x_fake, x2_target_lm)
    loss_fake = adv_loss(out, 0)

    loss = loss_real + loss_fake + args.lambda_reg * loss_reg

    return loss, Munch(real=loss_real.item(),
                       fake=loss_fake.item(),
                       reg=loss_reg.item())

# def compute_g_loss(nets, args, x1_source, x2_target, x2_target_lm, arcface, conf, z_trgs=None, x_refs=None, masks=None):
def compute_g_loss(nets, args, x1_source, x1_source_lm, x2_target, x2_target_lm, arcface, z_trgs=None, x_refs=None, masks=None):

    # adversarial loss
    s_trg = nets.style_encoder(x1_source)

    x_fake = nets.generator(x2_target_lm, s_trg, masks=masks)
    out = nets.discriminator(x_fake, x2_target_lm)
    loss_adv = adv_loss(out, 1)

    # content reconstruction loss
    # s_pred = nets.style_encoder(x_fake)
    #     # loss_con = torch.mean(torch.abs(s_pred - s_trg))

    # cycle-consistency loss
    # x_fake_2 = nets.generator(x1_source_lm, s_pred, masks=masks)
    loss_cyc_1 = torch.mean(torch.abs(x_fake - x2_target))
    # loss_cyc_2 = torch.mean(torch.abs(x_fake_2 - x1_source))

    # loss_cyc = loss_cyc_1 + loss_cyc_2

    # loss_cyc = torch.mean(torch.abs(x_fake - x2_target))

    # ID loss

    x1_source = nn.functional.interpolate(x1_source[:, :, :, :], size=(112, 112), mode='bilinear')
    x_fake = nn.functional.interpolate(x_fake[:, :, :, :], size=(112, 112), mode='bilinear')
    x2_target = nn.functional.interpolate(x2_target[:, :, :, :], size=(112, 112), mode='bilinear')
    # x_fake_2 = nn.functional.interpolate(x_fake_2[:, :, :, :], size=(112, 112), mode='bilinear')


    arcface.eval()
    with torch.torch.no_grad():
        source_embs = arcface(x1_source)


        target_embs = arcface(x2_target)
        fake_embs = arcface(x_fake)
        # fake2_embs = arcface(x_fake_2)
    cos = nn.CosineSimilarity(dim=1,eps=1e-6)
    output = cos(target_embs, fake_embs)
    loss_id_1 = torch.mean(1-output)


    # output2 = cos(source_embs, target_embs)
    # loss_id_2 = torch.mean(1-output2)
    #
    # print(loss_id_1)
    # print(loss_id_2)
    # assert False

    loss = loss_adv  + args.lambda_cyc * loss_cyc_1 + args.lambda_id * loss_id_1


    # return loss, Munch(adv=loss_adv.item(),
    #                    cyc_1=loss_cyc_1.item(),
    #                    cyc_2=loss_cyc_2.item(),
    #                    con=loss_con.item(),
    #                    id_1=loss_id_1.item(),
    #                    id_2=loss_id_2.item())
    return loss, Munch(adv=loss_adv.item(),
                       cyc_1=loss_cyc_1.item(),
                       id_1=loss_id_1.item())




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