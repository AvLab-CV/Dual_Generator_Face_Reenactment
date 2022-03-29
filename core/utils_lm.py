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
import network


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




@torch.no_grad()
def translate_and_reconstruct_sample(nets, args, x1, x2_target, x2_target_lm, filename, conf, arcface):
    N, C, H, W = x1.size()
    # with torch.no_grad():

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
def translate_and_reconstruct_multi(nets, args,x1_1_source, x1_2_source, x1_3_source, x1_4_source, x1_5_source, x1_6_source, x1_7_source, x1_8_source, x9_target, x9_target_lm, filename):
    N, C, H, W = x1_1_source.size()

    s_ref = nets.style_encoder(x1_1_source)
    s_ref += nets.style_encoder(x1_2_source)
    s_ref += nets.style_encoder(x1_3_source)
    s_ref += nets.style_encoder(x1_4_source)
    s_ref += nets.style_encoder(x1_5_source)
    s_ref += nets.style_encoder(x1_6_source)
    s_ref += nets.style_encoder(x1_7_source)
    s_ref += nets.style_encoder(x1_8_source)

    s_ref_mean = s_ref/8

    # s_ref = nets.style_encoder(x1)
    masks = None
    x_fake = nets.generator(x9_target_lm, s_ref_mean, masks=masks, loss_select=args.loss)

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
    x_concat = [x9_target_lm, x1_1_source, x1_2_source, x1_3_source, x1_4_source, x1_5_source, x1_6_source, x1_7_source, x1_8_source, x9_target,x_fake[:,[0,1,2],:,:]]

    x_concat = torch.cat(x_concat, dim=0)
    save_image(x_concat, N, filename, loss = args.loss)
    # save_image(x_fake[:,[3],:,:], N, filename+'_lm.jpg')
    # assert False
    del x_concat



@torch.no_grad()
def translate_and_reconstruct(nets, args, x1, x2_target, x2_target_lm, filename, embedder):
    N, C, H, W = x1.size()
    if args.id_embed:

        s_fea = embedder(x1)
        s_ref = nets.mlp(s_fea)
    else:
        s_ref = nets.style_encoder(x1)
    # s_ref = nets.style_encoder(x1)
    masks = None
    x_fake = nets.generator(x2_target_lm, s_ref, masks=masks, loss_select=args.loss)

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
    x_concat = [x2_target_lm, x1,x_fake[:,[0,1,2],:,:], x2_target]

    x_concat = torch.cat(x_concat, dim=0)
    save_image(x_concat, N, filename, loss = args.loss)
    # save_image(x_fake[:,[3],:,:], N, filename+'_lm.jpg')
    # assert False
    del x_concat

@torch.no_grad()
def translate_and_reconstruct2(nets, args, x1,x1_lm, x2_target, x2_target_lm, filename):
    N, C, H, W = x1.size()
    s_ref = nets.style_encoder(x1)
    if args.masks:
        # masks = nets.fan.get_heatmap(x2_target_lm)
        masks = x2_target_lm
    else:
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
def debug_image(nets, args, inputs, step, embedder = None):
    # x_src, y_src = inputs.x_src, inputs.y_src
    # x_ref, y_ref = inputs.x_ref, inputs.y_ref
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

        device = inputs.x1.device
        N = inputs.x1.size(0)

        filename = ospj(args.sample_dir, '%06d_cycle_consistency.jpg' % (step))
        translate_and_reconstruct_multi(nets, args, x1_1_source, x1_2_source, x1_3_source, x1_4_source, x1_5_source, x1_6_source, x1_7_source, x1_8_source, x9_target, x9_target_lm, filename)


    else:
        x1, x2_target, x2_target_lm = inputs.x1, inputs.x2, inputs.x2_lm
        # x1_lm =  inputs.x_lm

        # x1, x_source_4_channel, x2_target, x2_target_lm = inputs.x1, inputs.x1_c, inputs.x2, inputs.x2_lm
        #
        # x_source_4_channel = nn.functional.interpolate(x_source_4_channel[:, :, :, :], size=(128, 128), mode='bilinear')

        device = inputs.x1.device
        N = inputs.x1.size(0)

        # translate and reconstruct (reference-guided)
        filename = ospj(args.sample_dir, '%06d_cycle_consistency.jpg' % (step))
        translate_and_reconstruct(nets, args, x1, x2_target, x2_target_lm, filename, embedder)

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