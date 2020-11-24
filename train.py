

from os import path as osp
import os
import sys
cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.append(osp.join(cur_dir, '.'))

import json
import torch
from tensorboardX import SummaryWriter
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from DRRender import MaskRenderer as Render

from apex import amp

import cv2
from collections import OrderedDict
import numpy as np
from nets import DMTL
from config import args, dt_folder, train_set, eval_set, k_fold
import config
from datasets import (
    DataCacheWrapper,
    hand_cmu,
    hand_FreiHAND,
    hand_HIU,
    hand_RHD,
    hand_STB,
    hand_dexter,
    unnormalize_image,
    POF_Generator_cuda,
    HM_Generator_cuda,
    HMExtractor,
    BONE_HIERARCHY, 
    NUM_KPS,
)

import torch.nn.functional as F
import math
import datetime
from utils import  (
    AverageMeter,
    draw_circle,
    Agent,
    save_obj,
    batch_proj,
    batch_rodrigues,
    copy_state_dict,
    deterministic_training_procedure,
    reduce_tensor,
    draw_backbone,
)

from mano.manopth.manolayer import ManoLayer
from scheduler import PolyScheduler

class DataPrefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_batch = next(self.loader)
        except StopIteration:
            self.next_batch = None
            return

        with torch.cuda.stream(self.stream):
            for k in self.next_batch.keys():
                self.next_batch[k] = self.next_batch[k].cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        next_batch = self.next_batch
        self.preload()
        return next_batch

class Trainer(object):
    def __init__(self, rank=0, world_size=1, mixed_precision=False):
        self.device_count, self.rank, self.world_size, self.mixed_precision = torch.cuda.device_count(), rank, world_size, mixed_precision
        self.device = torch.device('cuda:{}'.format(self.rank)) if torch.cuda.is_available() else torch.device('cpu')
        self.__init_multi_gpu__()
        self._create_dataset()
        self._create_model()
        self.logger   = SummaryWriter('runs/pretrain')
        self.glb_iter = 0

    def __init_multi_gpu__(self):
        if self.device_count < 2:
            return
        torch.cuda.set_device(self.rank)
        dist.init_process_group(
            backend='nccl', 
            init_method='tcp://127.0.0.1:{}'.format(args.nccl_port), 
            world_size=self.world_size, 
            rank=self.rank
        )

    @staticmethod
    def to_half_buffer(module):
        for name, buf in module.named_buffers():
            if 'float' in str(type(buf)):
                module.named_buffers[name] = buf.half()

    def _create_model(self):
        cfg = config.cfg
        print(cfg)
        model =DMTL(**cfg, n_feats= 10+args.n_pca+3+args.n_cam).to(self.device)
        mano = ManoLayer(mano_root=osp.join(cur_dir, 'mano/mano/models'), use_pca=True, ncomps=args.n_pca, side='left', flat_hand_mean=False, to_milimeters = False)
        model = Agent(mano=mano, model=model).to(self.device).float()
        self.optimizer = torch.optim.Adam(model.model.parameters(), args.lr, weight_decay=args.wd)
        self.render = Render(self.device, faces=mano.th_faces.to(self.device), batch_size=args.batch_size, img_size=args.crop_size).to(self.device)

        #mixed precision
        if self.mixed_precision:
            model, self.optimizer = amp.initialize(model, self.optimizer, opt_level='O1')

        #parallel training
        if self.device_count <= 1:
            self.model = nn.DataParallel(model)
        else:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model).to(self.device)
            self.model = DDP(model, device_ids=[self.rank], output_device=self.rank, broadcast_buffers=False, find_unused_parameters=True)

        self.model = self.model.to(self.device)
        if args.use_cuda:
            self.pof_gen   = POF_Generator_cuda(**cfg, pof_conns=np.array(BONE_HIERARCHY)).to(self.device) if args.gen_pof else None
            self.hm_gen    = HM_Generator_cuda(**cfg).to(self.device) if args.gen_hm else None
        
        #2d pose extractor, see Equation 1 in the main paper
        self.hm_extractor = HMExtractor(hm_size=[128, 128], num_kp=NUM_KPS).float().to(self.device).eval()

        epoch_iter = math.ceil(self.train_samples / args.batch_size / self.device_count)
        total_iter = args.epoch * epoch_iter

        print('[sample: {}, total_iters:{}, iter_per_epoch: {}]'.format(self.train_samples, total_iter, epoch_iter))
        self.scheduler = PolyScheduler(self.optimizer, base_lr=args.lr,max_iters=total_iter,power=0.9,warmup_ratio=0.0)

        if args.resume and osp.isfile(args.check_point):
            print('load pretrained model {}'.format(args.check_point))
            pretrained = torch.load(args.check_point, map_location='cpu')
            copy_state_dict(self.model.module.model.state_dict(), pretrained, prefix='model.')

    def _create_dataset(self):
        dt_sets = []
        self.train_samples = 0
        for dt_name in train_set:
            dt_set = eval(dt_name)(dt_folder[dt_name],k_fold=k_fold[dt_name],mode='train',crop_size=args.crop_size,use_cuda=args.use_cuda,gen_hm=args.gen_hm, gen_pof=args.gen_pof,**config.cfg)
            print('dataset {} total {} training samples'.format(dt_name, len(dt_set)))
            dt_sets.append(
                DataCacheWrapper(
                    data_set=dt_set,
                    cache_prob=args.cache_prob,
                    cache_cap=args.cache_cap,
                )
            )
            self.train_samples += len(dt_set)
        
        train_dataset = torch.utils.data.ConcatDataset(dt_sets)
        self.train_sampler = DistributedSampler(train_dataset) if self.device_count > 1 else None
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.workers,
            shuffle = self.device_count < 2,
            pin_memory=True,
            sampler=self.train_sampler
        )
        self.train_loader, self.train_dataset = train_loader, train_dataset

        dt_sets = []
        for dt_name in eval_set:
            dt_set = eval(dt_name)(dt_folder[dt_name],mode='test', crop_size=args.crop_size,use_cuda=args.use_cuda,gen_hm=args.gen_hm, gen_pof=args.gen_pof,**config.cfg)
            print('dataset {} total {} testing samples'.format(dt_name, len(dt_set)))
            dt_sets.append(dt_set)
        
        test_dataset = torch.utils.data.ConcatDataset(dt_sets)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            num_workers=args.workers,
            shuffle=False,
            pin_memory=True,
            sampler=None,
        )
        self.test_loader, self.test_dataset = test_loader, test_dataset

    def batch_landmark_2d_loss(self, p_2d, R):
        r_2d = R['key_points_2d']
        mask = (r_2d[..., 2] < 2) * R['w_key_points_2d']
        return (((p_2d-r_2d[...,:2])**2).sum(-1).sqrt()*mask).sum()/(mask.sum()+1e-6)

    def batch_bone_rotate_loss(self, p_3d, R):
        r_2d, r_3d, w_3d = R['key_points_2d'], R['key_points_3d'], R['w_key_points_3d']
        pt_valid_2d, pt_valid_3d = r_2d[..., 2], r_3d[..., 3]
        mask = (pt_valid_2d[:, BONE_HIERARCHY[:, 0]] < 3) * (pt_valid_2d[:, BONE_HIERARCHY[:, 1]] < 3) * \
            ((pt_valid_2d[:, BONE_HIERARCHY[:, 0]] < 2 ) + (pt_valid_2d[:, BONE_HIERARCHY[:, 1]] < 2)) * \
            (pt_valid_3d[:, BONE_HIERARCHY[:, 0]] < 1) * (pt_valid_3d[:, BONE_HIERARCHY[:, 1]] < 1) *\
            w_3d[:, BONE_HIERARCHY[:, 1]]
        r_vec = r_3d[:, BONE_HIERARCHY[:,0],:3] - r_3d[:, BONE_HIERARCHY[:, 1], :3]
        p_vec = p_3d[:, BONE_HIERARCHY[:,0],:3] - p_3d[:, BONE_HIERARCHY[:, 1], :3]
        return ((1.0-F.cosine_similarity(r_vec, p_vec, dim=2, eps=1e-6))*mask).sum() / (mask.sum()+1e-6)

    def train(self, epoch):
        self.loss_hm         = AverageMeter()
        self.loss_pof        = AverageMeter()
        self.loss_mask       = AverageMeter()

        self.loss_sil        = AverageMeter()
        self.loss_2d         = AverageMeter()
        self.loss_3d_dir     = AverageMeter()
        
        self.self_sup_mask   = AverageMeter()
        self.self_sup_pose   = AverageMeter()

        self.reg_beta        = AverageMeter()
        self.reg_theta       = AverageMeter()
        self.avg_loss        = AverageMeter()

        optimizer, scheduler = self.optimizer, self.scheduler
        logger = self.logger

        if self.device_count > 1:
            self.train_sampler.set_epoch(epoch)
        prefetcher = DataPrefetcher(self.train_loader)
        R = prefetcher.next()

        self.model.eval()
        self.model.module.model.train()

        iter_index = -1

        while R is not None:
            self.glb_iter += 1
            loss_hm, loss_pof, loss_mask           = [], [], [] #training the backbone
            loss_2d, loss_sil, loss_3d_dir         = [], [], [] #training the regressor heads
            loss_self_sup_mask, loss_self_sup_2d   = [], []     #achieving self-supervised learning
            loss_reg_beta, loss_reg_theta          = [], []     #reg terms

            iter_index += 1

            I, r_2d, r_3d, r_mask  = R['image'], R['key_points_2d'], R['key_points_3d'], R['seg_mask']
            if args.use_cuda:
                with torch.no_grad():
                    (r_hm,  hm_valid)  = self.hm_gen(key_points_2d=r_2d) if args.gen_hm else (None, None)
                    (r_pof, pof_valid) = self.pof_gen(key_points_2d=r_2d, key_points_3d=r_3d) if args.gen_pof else (None, None)	
            else:
                (r_hm,  hm_valid)  = (R['hm'], R['hm_valid']) if args.gen_hm else (None, None)
                (r_pof, pof_valid) = (R['pof'], R['pof_valid']) if args.gen_pof else (None, None)

            mano_outputs, joints_2ds, hms, pofs, masks = self.model(I)
            for stage, (mano_output, p_2d, p_hm, p_pof, p_mask) in enumerate(zip(mano_outputs, joints_2ds, hms, pofs, masks)):
                p_3d, p_theta, p_beta = mano_output['joints'], mano_output['thetas'], mano_output['betas']

                #reg theta
                loss_reg_theta.append((p_theta[:,3:]**2).mean())

                #reg beta
                loss_reg_beta.append((p_beta**2).mean())

                #2d loss
                loss_2d.append(self.batch_landmark_2d_loss(p_2d=p_2d, R=R))

                #L2 heatmap loss
                mask = hm_valid
                w_hm = R['w_hm']
                mask = mask*w_hm.unsqueeze(-1)
                p_hm = F.upsample_bilinear(p_hm, size=r_hm.shape[2:])
                loss = (((r_hm-p_hm)**2).mean(dim=-1).mean(dim=-1)*mask).sum()/(mask.sum()+1e-6)
                loss_hm.append(loss)

                #L2 pof loss
                mask  = pof_valid
                w_pof = R['w_pof']
                mask  = mask*w_pof.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) #N x K x H x W
                p_pof = F.upsample_bilinear(p_pof, size=r_pof.shape[2:4])
                loss  = (((r_pof-p_pof)**2) * mask).sum()/(mask.sum()+1e-6)
                loss_pof.append(loss)

                #semantic mask loss
                mask = R['w_mask']
                p_mask = F.sigmoid(F.upsample_bilinear(p_mask, size=r_mask.shape[2:4]))
                loss = -(torch.log(p_mask+1e-6)*r_mask + torch.log(1.0-p_mask+1e-6)*(1.0-r_mask)).mean(-1).mean(-1).mean(-1)
                loss = (loss * mask).sum() / (mask.sum()+1e-6)
                loss_mask.append(loss)

                #3d bone_dir loss
                loss_3d_dir.append(self.batch_bone_rotate_loss(p_3d=p_3d, R=R))

                #re-projected hand mask
                p_verts = batch_proj(mano_output['verts'], mano_output['w_cam'], args.proj_type, keepz=True)
                p_sil   = self.render(p_verts, scale2opengl=True)
                sil_a, sil_b, mask = r_mask[:,0], p_mask[:,0], R['w_mask']
                loss_sil_a = ((sil_a-p_sil).abs().mean(-1).mean(-1)*mask).sum()/(mask.sum()+1e-6)
                loss_sil.append(loss_sil_a)

                #self-supervised hand mask loss
                w_ssl = R['w_ssl']
                mask  = w_ssl
                loss_sil_b = ((sil_b-p_sil).abs().mean(-1).mean(-1)*mask).sum()/(mask.sum()+1e-6)
                loss_self_sup_mask.append(loss_sil_b)

                #self-supervised 2d hand pose loss
                p_2d_b = self.hm_extractor(p_hm)
                mask = w_ssl.unsqueeze(-1)
                loss_self_sup_pose = (((p_2d-p_2d_b[...,:2])**2).sum(-1).sqrt()*mask).sum()/(mask.sum()+1e-6)
                loss_self_sup_2d.append(loss_self_sup_pose)
                
            stage = len(mano_outputs)
            loss_hm, loss_pof, loss_mask         = sum(loss_hm)/stage, sum(loss_pof)/stage, sum(loss_mask)/stage
            loss_2d, loss_3d_dir, loss_sil       = sum(loss_2d)/stage, sum(loss_3d_dir)/stage, sum(loss_sil)/stage
            loss_self_sup_mask, loss_self_sup_2d = sum(loss_self_sup_mask)/stage, sum(loss_self_sup_2d)/stage
            loss_reg_beta, loss_reg_theta        = sum(loss_reg_beta)/stage, sum(loss_reg_theta)/stage

            loss = loss_hm*args.w_hm  + loss_pof*args.w_pof + loss_mask*args.w_mask + \
                loss_2d*args.w_2d + loss_3d_dir*args.w_3d_dir + loss_sil*args.w_silhouette + \
                loss_self_sup_2d*args.w_self_sup_2d + loss_self_sup_mask*args.w_self_sup_mask + \
                loss_reg_beta*args.w_reg_beta + loss_reg_theta*args.w_reg_theta
                
            optimizer.zero_grad()
            if self.mixed_precision:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            
            #clip the convolutional weights.
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
            optimizer.step()

            lr       = scheduler.step()
            R        = prefetcher.next()


            self.loss_hm.update(float(loss_hm))
            self.loss_pof.update(float(loss_pof))
            self.loss_mask.update(float(loss_mask))
                
            self.loss_sil.update(float(loss_sil))
            self.loss_2d.update(float(loss_2d))
            self.loss_3d_dir.update(float(loss_3d_dir))

            self.self_sup_mask.update(float(loss_self_sup_mask))
            self.self_sup_pose.update(float(loss_self_sup_2d))
            
            self.reg_beta.update(float(loss_reg_beta))
            self.reg_theta.update(float(loss_reg_theta))

            self.avg_loss.update(float(loss))

            if self.rank != 0:
                continue
    
            msg = OrderedDict([
                ('time',datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')),
                ('epoch',epoch),
                ('iter',iter_index),
                ('lr',lr),
                ('loss', round(self.avg_loss.average(), 8)),

                ('loss_hm', round(self.loss_hm.average(), 8)),
                ('loss_pof', round(self.loss_pof.average(), 8)),
                ('loss_mask', round(self.loss_mask.average(), 8)),
                ('loss_sil', round(self.loss_sil.average(), 8)),
                ('loss_2d', round(self.loss_2d.average(), 8)),
                ('loss_3d_dir', round(self.loss_3d_dir.average(), 8)),
                ('loss_self_sup_pose', round(self.self_sup_pose.average(), 8)),
                ('loss_self_sup_mask', round(self.self_sup_mask.average(), 8)),
                ('reg_beta', round(self.reg_beta.average(), 8)),
                ('reg_theta', round(self.reg_theta.average(), 8)),
            ])

            for (k, v) in msg.items():
                if k not in['time', 'epoch', 'iter', 'lr']:
                    logger.add_scalar(k, v, global_step=self.glb_iter)

            print(msg)
            sys.stdout.flush()

    def do_train(self):
        check_folder = './checkpoint'
        os.makedirs(check_folder, exist_ok=True)

        for epoch in range(args.epoch):
            if self.rank==0 and epoch % 4 == 0:
                torch.save(self.model.module.state_dict(), osp.join(check_folder, '{}.pth'.format(epoch)))
            self.train(epoch)

def worker(rank, world_size, *args, **kargs):
    trainer = Trainer(rank=rank, world_size=world_size, mixed_precision=False)
    trainer.do_train()

def main():
    print(args)
    deterministic_training_procedure(args.seed)
    world_size = torch.cuda.device_count()
    if world_size <= 1:
        worker(0, 1)
    else:
        mp.spawn(worker, nprocs=world_size, args=(world_size,))

if __name__ == '__main__':
    main()
